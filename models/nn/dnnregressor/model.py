import tensorflow as tf
import numpy as np
import pandas as pd
import dask.dataframe as dd

tf.logging.set_verbosity(tf.logging.INFO)

# Cloud storage bucket where the data lives.
BUCKET = 'gs://nyc-taxi-fare-prediction-data/'

# Columns in the train/dev sets.
CSV_COLUMNS = ['fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude',
               'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'distance',
               'duration', 'summary', 'same_pickup_dropoff', 'year', 'month', 'dow',
               'hour', 'pickup_latitude_quantile', 'dropoff_latitude_quantile',
               'pickup_longitude_quantile', 'dropoff_longitude_quantile',
               'pickup_distance_to_jfk', 'pickup_jfk', 'dropoff_distance_to_jfk',
               'dropoff_jfk', 'pickup_distance_to_lga', 'pickup_lga',
               'dropoff_distance_to_lga', 'dropoff_lga', 'pickup_distance_to_ewr',
               'pickup_ewr', 'dropoff_distance_to_ewr', 'dropoff_ewr', 'summary1',
               'summary2', 'bearing_degrees', 'bearing_bucket', 'haversine_distance']

# Columns in the test set.
CSV_COLUMNS_TEST = ['key'] + CSV_COLUMNS[1:]

# Column of the response.
LABEL_COLUMN = 'fare_amount'

# Frequencies for top summaries.
s1_freq = pd.read_csv('../data/summary1_freq.csv')
s2_freq = pd.read_csv('../data/summary2_freq.csv')

# Compute mean/variance of distance and duration to enable feature scaling.
dev = dd.read_csv(BUCKET + 'dev.csv').compute()
distance_stats = {'mean': dev.distance.mean(),
                  'sd': np.sqrt(dev.distance.var())}
duration_stats = {'mean': dev.duration.mean(),
                  'sd': np.sqrt(dev.duration.var())}
def normalize(x, mean, sd):
    return (x - mean) / sd

# Non float defaults.
DEFAULTS = {
    'key': [''],
    'pickup_datetime': [''],
    'summary': [''],
    'summary1': [''],
    'summary2': [''],
    'bearing_bucket': ['None'],
    'year': [2008],
    'month': [0],
    'dow': [0],
    'hour': [0],
    'pickup_latitude_quantile': [0],
    'dropoff_latitude_quantile': [0],
    'pickup_longitude_quantile': [0],
    'dropoff_longitude_quantile': [0],
    'pickup_jfk': [0],
    'dropoff_jfk': [0],
    'pickup_lga': [0],
    'dropoff_lga': [0],
    'pickup_ewr': [0],
    'dropoff_ewr': [0]
}


def read_dataset(filename, mode, batch_size=512):
    def _input_fn():
        def decode_csv(value_column):
            if mode == tf.estimator.ModeKeys.PREDICT:
                expected_columns = CSV_COLUMNS_TEST
            else:
                expected_columns = CSV_COLUMNS
            columns = tf.decode_csv(
                value_column,
                record_defaults=[[] if col not in DEFAULTS else DEFAULTS[col] for col in expected_columns])
            features = dict(zip(expected_columns, columns))
            label = features.pop(LABEL_COLUMN, -1)
            return features, label
        full_path = BUCKET + filename
        dataset = tf.data.TextLineDataset(full_path).skip(1).map(decode_csv)
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None
            dataset = dataset.shuffle(buffer_size=10 * batch_size)
        else:
            num_epochs = 1

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()

    return _input_fn

pickup_columns = [
    tf.feature_column.categorical_column_with_identity(
        key='pickup_latitude_quantile', num_buckets=10),
    tf.feature_column.categorical_column_with_identity(
        key='pickup_longitude_quantile', num_buckets=10)
]

dropoff_columns = [
    tf.feature_column.categorical_column_with_identity(
        key='dropoff_latitude_quantile', num_buckets=10),
    tf.feature_column.categorical_column_with_identity(
        key='dropoff_longitude_quantile', num_buckets=10)
]

time_columns = [
    tf.feature_column.categorical_column_with_identity('dow', num_buckets=7),
    tf.feature_column.categorical_column_with_identity('hour', num_buckets=24)
]

INPUT_COLUMNS = [
    tf.feature_column.embedding_column(tf.feature_column.crossed_column(
        pickup_columns, 100), 10),
    tf.feature_column.embedding_column(tf.feature_column.crossed_column(
        dropoff_columns, 100), 10),
    tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'year', vocabulary_list=range(2008, 2016)), 10),
    tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'month', vocabulary_list=range(1, 13)), 5),
    tf.feature_column.embedding_column(tf.feature_column.crossed_column(
        time_columns, 7*24), 10),
    tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'bearing_bucket',
            vocabulary_list=['E', 'N', 'NE', 'NW', 'None', 'S', 'SE', 'SW', 'W']),
        10),
    tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'summary1',
            vocabulary_list=['Other'] + s1_freq.label[:20].fillna('None').values,
            default_value=0),
        10),
    tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'summary2',
            vocabulary_list=['Other'] + s2_freq.label[:20].fillna('None').values,
            default_value=0),
        10),
    tf.feature_column.numeric_column(
        'distance', normalizer_fn=lambda x: normalize(x, **distance_stats)),
    tf.feature_column.numeric_column(
        'duration', normalizer_fn=lambda x: normalize(x, **duration_stats)),
    tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_identity('pickup_jfk', num_buckets=2)),
    tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_identity('dropoff_jfk', num_buckets=2)),
    tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_identity('pickup_lga', num_buckets=2)),
    tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_identity('dropoff_lga', num_buckets=2)),
    tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_identity('pickup_ewr', num_buckets=2)),
    tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_identity('dropoff_ewr', num_buckets=2))
]

def train_and_evaluate(args):
    runconfig = tf.estimator.RunConfig(model_dir=args['output_dir'], keep_checkpoint_max=1, 
                                       save_summary_steps=25000, log_step_count_steps=25000,
                                       save_checkpoints_steps=10000)
    estimator = tf.estimator.DNNRegressor(hidden_units = args['hidden_units'],
                                          feature_columns = INPUT_COLUMNS,
                                          model_dir = args['output_dir'])
    
    train_spec=tf.estimator.TrainSpec(
        input_fn = read_dataset('train.csv', mode = tf.estimator.ModeKeys.TRAIN),
        max_steps = args['train_steps'])

    eval_spec=tf.estimator.EvalSpec(
        input_fn = read_dataset('dev.csv', mode = tf.estimator.ModeKeys.EVAL),
        steps = None,
        throttle_secs=300)
    
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    
    test = dd.read_csv(BUCKET + 'test.csv').compute()
    preds = estimator.predict(
        input_fn = read_dataset('test.csv', mode = tf.estimator.ModeKeys.PREDICT))
    pd.DataFrame({
        'fare_amount': [pred['predictions'][0] for pred in preds],
        'key': test.key.values
    }).to_csv('prediction.csv', index=False)
        
