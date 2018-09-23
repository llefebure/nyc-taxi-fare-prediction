'''Train and validate an ExtraTreesRegressor model'''
import pandas as pd
import dask.dataframe as dd
import numpy as np
import gcsfs
import re
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

dtypes = {
    'year': 'str',
    'month': 'str',
    'hour': 'str',
    'dow': 'str',
    'same_pickup_dropoff': 'str',
    'pickup_latitude_quantile': 'str',
    'pickup_longitude_quantile': 'str',
    'dropoff_latitude_quantile': 'str',
    'dropoff_longitude_quantile': 'str',
    'pickup_jfk': 'str',
    'dropoff_jfk': 'str'
}

EXPECTED_NUM_ROWS = 53689806
BATCH_SIZE = 1000000
EXPECTED_FOREST_COUNT = 250

expected_num_batches = np.int(np.floor(EXPECTED_NUM_ROWS / BATCH_SIZE))
n_estimators_per_batch = np.int(
    np.floor(EXPECTED_FOREST_COUNT / expected_num_batches))

print('Expected number of batches: %d' % expected_num_batches, flush=True)
print('Number of estimators per batch: %d' % n_estimators_per_batch, flush=True)

# Define features to include in model
features = ['distance', 'duration', 'year', 'month', 'dow', 'hour', 'bearing_bucket',
            'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
            'dropoff_longitude', 'pickup_jfk', 'dropoff_jfk', 'summary1', 'summary2',
            'same_pickup_dropoff']

# Set levels of categorical variables (infer from full dev set)
dev = (dd.read_csv('gs://nyc-taxi-fare-prediction-data/dev.csv', dtype=dtypes)
       .compute())
def convert_to_categoricals(df):
    top_roads = ['Queens-Midtown Tunnel', 'FDR Drive', 'Park Avenue',
                 'Madison Avenue', 'ith Avenue', 'Lexington Avenue',
                 'Central Park West']
    for col in dtypes:
        if dtypes[col] == 'str':
            df[col] = pd.Categorical(df[col], dev[col].unique())
    ith_avenue_pattern = r'[0-9]{1,2}(th|nd|rd|st) Avenue'
    def _encode_summary(x):
        if type(x) == str and re.match(ith_avenue_pattern, x):
            return 'ith Avenue'
        else:
            return x
    df['summary1'] = pd.Categorical(
        df.summary1.apply(_encode_summary), top_roads)
    df['summary2'] = pd.Categorical(
        df.summary2.apply(_encode_summary), top_roads)

convert_to_categoricals(dev)
dev_transformed = pd.get_dummies(dev[features])

# Main model training loop
model = ExtraTreesRegressor(n_estimators=n_estimators_per_batch,
                            n_jobs=6, warm_start=True)
fs = gcsfs.GCSFileSystem(project='steadfast-mason-213717')
with fs.open('nyc-taxi-fare-prediction-data/train.csv') as f:
    i = 0
    for chunk in pd.read_csv(f, dtype=dtypes, chunksize=BATCH_SIZE):
        convert_to_categoricals(chunk)
        chunk_transformed = pd.get_dummies(chunk[features])
        model.fit(chunk_transformed, chunk['fare_amount'])
        model.n_estimators += n_estimators_per_batch
        rmse = np.sqrt(mean_squared_error(
            model.predict(dev_transformed), dev['fare_amount']))
        print('RMSE after %.2f%% of training data: %.4f' % 
              (100. * i / expected_num_batches, rmse), flush=True)
        i += 1

# Feature importance
print('\nFeature Importance\n----------------')
for f, imp in sorted(zip(dev_transformed.columns, model.feature_importances_), 
                     key=lambda x: x[1], reverse=True):
    print('%s: %.4f' % (f, imp), flush=True)

# Test predictions
test = (dd.read_csv('gs://nyc-taxi-fare-prediction-data/test.csv', dtype=dtypes)
        .compute())
convert_to_categoricals(test)
test_transformed = pd.get_dummies(test[features])
test_preds = pd.DataFrame({
    'key': test.key,
    'fare_amount': model.predict(pd.get_dummies(test_transformed))
})
test_preds.to_csv('prediction.csv', index=False)
