import pandas as pd
import tensorflow as tf

train_data = pd.read_csv('data/train.csv', index_col=0)
test_data = pd.read_csv('data/test.csv', index_col=0)
all_data = pd.concat([train_data, test_data])


# normalize numerical data
def normalize(train, test):
    numerical_cols = ['x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6']
    all = pd.concat([train, test])
    normalized_data = (all - all.min()) / (all.max() - all.min())
    normalized_train = normalized_data.iloc[0:len(train)]
    normalized_test = normalized_data.iloc[len(train):]

    train[numerical_cols] = normalized_train[numerical_cols]
    test[numerical_cols] = normalized_test[numerical_cols]


# normalize continous data
normalize(train=train_data, test=test_data)


def one_hot_encoding(data, all_data):
    # create one-hot encoding
    num_days = all_data['Day'].max()
    num_markets = all_data['Market'].max()
    num_stocks = all_data['Stock'].max()

    one_hot_days = tf.one_hot(data['Day'], depth=num_days)
    one_hot_markets = tf.one_hot(data['Market'], depth=num_markets)
    one_hot_stocks = tf.one_hot(data['Stock'], depth=num_stocks)

    return one_hot_days, one_hot_markets, one_hot_stocks


one_hot_days, one_hot_markets, one_hot_stocks = one_hot_encoding(train_data, all_data=all_data)

train_data.head()
