import tensorflow as tf
import pandas as pd

COLUMNS = ['pkSeqID', 'proto', 'saddr', 'sport', 'daddr',
                   'dport', 'seq', 'stddev', 'N_IN_Conn_P_SrcIP',
                   'min', 'state_number', 'mean', 'N_IN_Conn_P_DstIP',
                   'drate', 'srate', 'max', 'attack', 'category', 'subcategory']

# Load dataset.
df_train = pd.read_csv('/home/john/Downloads/UNSW_2018_IoT_Botnet_Final_10_best_Training_short.csv', header=0)
df_test = pd.read_csv('/home/john/Downloads/UNSW_2018_IoT_Botnet_Final_10_best_Testing_short.csv', header=0)


# Show datatypes
print(df_train.shape, df_test.shape)
print(df_train.dtypes)

# Show
print(df_train["attack"].value_counts())
### The model will be correct in atleast 70% of the case
print(df_test["attack"].value_counts())
## Unbalanced label
print(df_train.dtypes)

## Add features to the bucket:
### Define continuous list
NUMERIC_FEATURES = ['pkSeqID', 'saddr', 'sport', 'daddr',
                   'dport', 'seq', 'stddev', 'N_IN_Conn_P_SrcIP',
                   'min', 'state_number', 'mean', 'N_IN_Conn_P_DstIP',
                   'drate', 'srate', 'max']
### Define the categorical list
CATEGORICAL_FEATURES = ['proto', 'category', 'subcategory']

continuous_features = [tf.feature_column.numeric_column(k) for k in NUMERIC_FEATURES]

categorical_features = [tf.feature_column.categorical_column_with_hash_bucket(k, hash_bucket_size=1000) for k in CATEGORICAL_FEATURES]

model = tf.estimator.LinearClassifier(
    n_classes = 2,
    model_dir="ongoing/train",
    feature_columns=categorical_features + continuous_features)

FEATURES = ['pkSeqID', 'proto', 'saddr', 'sport', 'daddr',
            'dport', 'seq', 'stddev', 'N_IN_Conn_P_SrcIP',
            'min', 'state_number', 'mean', 'N_IN_Conn_P_DstIP',
            'drate', 'srate', 'max', 'category', 'subcategory']
LABEL= 'attack'
def get_input_fn(data_set, num_epochs=None, n_batch = 128, shuffle=True):
    return tf.compat.v1.estimator.inputs.pandas_input_fn(
       x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
       y = pd.Series(data_set[LABEL].values),
       batch_size=n_batch,
       num_epochs=num_epochs,
       shuffle=shuffle)

model.train(input_fn=get_input_fn(df_train,
            num_epochs=None,
            n_batch = 128,
            shuffle=False),
            steps=1000)

model.evaluate(input_fn=get_input_fn(df_test,
            num_epochs=1,
            n_batch = 128,
            shuffle=False),
            steps=1000)
