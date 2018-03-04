import pandas as pd
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

NUMERICAL_COLS = ['x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6']
NUM_HIDDEN_UNITS = [2048, 512, 256, 128, 64]


# normalize numerical data
def normalize_data(train, test):
    all_data = pd.concat([train, test])
    normalized_data = (all_data - all_data.min()) / (all_data.max() - all_data.min())
    normalized_train = normalized_data.iloc[0:len(train)]
    normalized_test = normalized_data.iloc[len(train):]

    train[NUMERICAL_COLS] = normalized_train[NUMERICAL_COLS]
    test[NUMERICAL_COLS] = normalized_test[NUMERICAL_COLS]


def g_model_fn(features, labels, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.name_scope('input'):
        days_one_hot = tf.one_hot(features['days'], depth=params['num_days'])
        markets_one_hot = tf.one_hot(features['markets'], depth=params['num_markets'])
        stocks_one_hot = tf.one_hot(features['stocks'], depth=params['num_stocks'])

        net = tf.concat([days_one_hot, markets_one_hot, stocks_one_hot, features['numerical_input']], 1)

    with tf.name_scope('net'):
        for hidden_unit in NUM_HIDDEN_UNITS:
            net = tf.layers.dense(inputs=net, units=hidden_unit, activation=tf.nn.relu)
            net = tf.layers.dropout(inputs=net, rate=params['dropout_rate'], training=is_training)
        predictions = tf.layers.dense(inputs=net, units=1)

    # check if inference
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    with tf.name_scope('loss'):
        # weighted MSE
        loss = tf.reduce_mean(tf.pow(labels - predictions, 2) * features['weights'], name='loss')

    metrics = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions)
    }

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)

    # Add evaluation metrics (for EVAL mode)
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)


def main(argv):
    train_data = pd.read_csv('data/train.csv', index_col=0).fillna(method='ffill')
    test_data = pd.read_csv('data/test.csv', index_col=0).fillna(method='ffill')
    all_data = pd.concat([train_data, test_data])

    model_num = 3

    # normalize continous data
    normalize_data(train=train_data, test=test_data)

    # create one-hot encoding
    params = {
        'num_days': all_data['Day'].max(),
        'num_markets': all_data['Market'].max(),
        'num_stocks': all_data['Stock'].max(),
        'dropout_rate': 0.4
    }

    input_train = {
        'days': train_data['Day'].values,
        'markets': train_data['Market'].values,
        'stocks': train_data['Stock'].values,
        'weights': train_data['Weight'].values.astype(np.float32),
        'numerical_input': train_data[NUMERICAL_COLS].values.astype(np.float32)
    }

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=input_train,
        y=train_data['y'].values.astype(np.float32),
        batch_size=100,
        num_epochs=None,
        shuffle=False)

    # Create the Estimator
    g_regression_model = tf.estimator.Estimator(model_fn=g_model_fn,
                                                params=params,
                                                model_dir='/tmp/g_model/{}'.format(model_num))

    # train the estimator
    g_regression_model.train(input_fn=train_input_fn, steps=317000)

    # perform inference
    input_predict = {
        'days': test_data['Day'].values,
        'markets': test_data['Market'].values,
        'stocks': test_data['Stock'].values,
        'numerical_input': test_data[NUMERICAL_COLS].values.astype(np.float32)
    }
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x=input_predict,
                                                          num_epochs=1,
                                                          shuffle=False)
    predictions = g_regression_model.predict(input_fn=predict_input_fn)
    predictions = [p[0] for p in predictions]

    # prepare submission
    submission = pd.DataFrame(predictions, columns=['y'])
    submission.index.name = 'Index'
    submission.to_csv('submissions/submission_{}.csv'.format(model_num))


if __name__ == "__main__":
    tf.app.run(main)
