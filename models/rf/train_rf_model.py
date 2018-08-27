import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
EXPECTED_FOREST_COUNT = 500

expected_num_batches = EXPECTED_NUM_ROWS / BATCH_SIZE
n_estimators_per_batch = EXPECTED_FOREST_COUNT / expected_num_batches

print 'Expected Number of Batches: ', expected_num_batches
print 'Number Estimators per Batch: ', n_estimators_per_batch

# Define features to include in model
features = ['distance', 'duration', 'year', 'month', 'dow', 'hour', 'bearing_bucket',
            'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',
            'pickup_jfk', 'dropoff_jfk']

# Set levels of categorical variables (infer from full dev set)
dev = pd.read_csv('../data/dev.csv', dtype=dtypes)
def convert_to_categoricals(df):
  for col in dtypes:
    if dtypes[col] == 'str':
        df[col] = pd.Categorical(df[col], dev[col].unique())

convert_to_categoricals(dev)
dev_transformed = pd.get_dummies(dev[features])

# Main model training loop
model = RandomForestRegressor(n_estimators=n_estimators_per_batch, n_jobs=-1, warm_start=True)
for i, chunk in enumerate(pd.read_csv('../data/train.csv', dtype=dtypes, chunksize=BATCH_SIZE)):
  print 100. * i / expected_num_batches, '%'
  convert_to_categoricals(chunk)
  chunk_transformed = pd.get_dummies(chunk[features])
  model.fit(pd.get_dummies(chunk[features]), chunk['fare_amount'])
  model.n_estimators += n_estimators_per_batch
  print 'RMSE: ', np.sqrt(mean_squared_error(model.predict(dev_transformed), dev['fare_amount']))

# Test predictions
test = pd.read_csv('../data/test.csv', dtype=dtypes)
convert_to_categoricals(test)
test_transformed = pd.get_dummies(test[features])
# print set(dev_transformed.columns) - set(test_transformed.columns)
test_transformed['year_2008'] = 0
test_preds = model.predict(test_transformed)
pd.DataFrame({'key': test.key, 'fare_amount': test_preds}).to_csv('prediction.csv', index=False)