import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from IPython.display import clear_output
from matplotlib import pyplot as plt


# Load dataset.
dftrain = pd.read_csv('/home/john/Downloads/UNSW_2018_IoT_Botnet_Final_10_best_Training.csv')
dfeval = pd.read_csv('/home/john/Downloads/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv')
y_train = dftrain.pop('attack')
y_eval = dfeval.pop('attack')

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


# It processes in batches since this is such a huge dataset.

def make_input_fn(X, y):
    def input_fn():
        dataset = tf.data.experimental.make_csv_dataset(
            (dict(X), y),
            batch_size=200, # 200 worked pretty good.
            na_value="?",
            num_epochs=1,
            shuffle=False,
            ignore_errors=False)
        return dataset
    return input_fn


train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval)



linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(pd.Series(result))
