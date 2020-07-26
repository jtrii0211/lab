import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from IPython.display import clear_output
from matplotlib import pyplot as plt
import seaborn as sns
sns_colors = sns.color_palette('colorblind')



# Load dataset.
dftrain = pd.read_csv('/home/john/Downloads/UNSW_2018_IoT_Botnet_Final_10_best_Training.csv')
dfeval = pd.read_csv('/home/john/Downloads/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv')
y_train = dftrain.pop('attack')
y_eval = dfeval.pop('attack')


fc = tf.feature_column
CATEGORICAL_COLUMNS = ['proto', 'saddr', 'sport', 'daddr', 'dport', 'category', 'subcategory']
NUMERIC_COLUMNS = ['pkSeqID', 'seq', 'stddev', 'N_IN_Conn_P_SrcIP',
                   'min', 'state_number', 'mean', 'N_IN_Conn_P_DstIP',
                   'drate', 'srate', 'max']


feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


# Use chunks since this is such a huge dataset.

def make_input_fn(X, y):
    def input_fn():
        dataset = tf.data.experimental.make_csv_dataset(
            (dict(X), y),
            batch_size=1000, # 200 worked pretty good.
            na_value="?",
            num_epochs=1,
            shuffle=False,
            ignore_errors=True)
        return dataset
    return input_fn


train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval)


# Helps generate graphs of quality of features
params = {
  'n_trees': 50,
  'max_depth': 3,
  'n_batches_per_layer': 1,
  # You must enable center_bias = True to get DFCs. This will force the model to
  # make an initial prediction before using any features (e.g. use the mean of
  # the training labels for regression or log odds for classification when
  # using cross entropy loss).
  'center_bias': True
}

est = tf.estimator.BoostedTreesClassifier(feature_columns, **params)
# Train model.
est.train(train_input_fn, max_steps=100)
# Evaluation.
results = est.evaluate(eval_input_fn)

# Make predictions.
pred_dicts = list(est.experimental_predict_with_explanations(eval_input_fn))

# Create DFC Pandas dataframe.
labels = y_eval.values
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
df_dfc = pd.DataFrame([pred['dfc'] for pred in pred_dicts])
df_dfc.describe().T
