Predicting Taxi Fares in NYC
============================

This is the code for a recent [Kaggle competition](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction). I used this competition to get more applied experience with neural networks and TensorFlow in general. The problem itself is pretty simple: given geocoordinates of pickup and dropoff and a timestamp, predict the fare amount.

My work here is divided into three components.

## Gathering External Data

My initial thought when approaching this problem was that we would want features about the route such as driving distance, duration, and street directions. After some research, I decided to get this data through [OSRM](https://github.com/Project-OSRM/osrm-backend) which is a C++ routing tool that can run offline on OpenStreetMap dumps. See the [feature_augmentation](feature_augmentation) subdirectory for more details about this effort.

## Data Cleaning and Feature Engineering

There were several issues with the training data such as obviously malformed geocoordinates and fare amount outliers, so I dealt with that. Additionally, I created several features such as coordinate quantile buckets and year, month, day of week, and hour from the timestamp. See [here](models/Data%20Preparation%20and%20Exploration.ipynb) for more details.

## Model Building

I started out with some simple [benchmarks](models/Benchmarks.ipynb) using linear regression and random forests on small subsets of the data. The benchmarks are quite good. For the full scale models, I experimented with two: the ExtraTreesRegressor from `scikit-learn` and the `DNNRegressor` from TensorFlow. See more details [here](models).