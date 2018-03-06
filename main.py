import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)

NUMERICAL_COLS = ['x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6']
MODELS_DIR = os.getenv('MODELS_DIR', '/tmp/models')


def main(argv):
    hidden_units = [
        [1024, 256],
        [512, 128],
        [256, 128],
        [512],
        [1024]
    ]

    for lr in [0.01, 0.001, 0.0001, 0.00001]:
        for dr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for hu in hidden_units:
                for shuffle in [False, True]:
                    execute(learning_rate=lr, dropout_rate=dr, hidden_units=hu, shuffle=shuffle)


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
        for hidden_unit in params['hidden_units']:
            net = tf.layers.dense(inputs=net, units=hidden_unit, activation=tf.nn.relu)
            net = tf.layers.dropout(inputs=net, rate=params['dropout_rate'], training=is_training)
        predictions = tf.layers.dense(inputs=net, units=1)

    # check if inference
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    with tf.name_scope('loss'):
        # weighted MSE
        loss_name = 'loss_eval' if mode == tf.estimator.ModeKeys.EVAL else 'loss'
        loss = tf.reduce_mean(tf.pow(labels - predictions, 2) * features['weights'], name=loss_name)

    metrics = {
        # "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions)
    }

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)

    # Add evaluation metrics (for EVAL mode)
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)


def execute(learning_rate, dropout_rate, hidden_units, shuffle):
    train_data = pd.read_csv('data/train.csv', index_col=0).fillna(method='ffill')
    test_data = pd.read_csv('data/test.csv', index_col=0).fillna(method='ffill')
    all_data = pd.concat([train_data, test_data])

    # normalize continous data
    normalize_data(train=train_data, test=test_data)

    # train_data = train_data.ix[:10]
    train_data, eval_data = train_test_split(train_data, test_size=0.2, shuffle=shuffle)

    # create hyperparameter
    params = {
        'num_days': all_data['Day'].max(),
        'num_markets': all_data['Market'].max(),
        'num_stocks': all_data['Stock'].max(),
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'hidden_units': hidden_units,
        'shuffle': shuffle
    }

    model_name = parameter_to_name(params)
    print('Start training with model {}'.format(model_name))

    # specify training
    batch_size = 100
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x=frame_to_dict(train_data),
                                                        y=train_data['y'].values.astype(np.float32),
                                                        batch_size=batch_size,
                                                        num_epochs=1,
                                                        shuffle=shuffle)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=120000)

    # specify validation
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x=frame_to_dict(eval_data),
                                                       y=eval_data['y'].values.astype(np.float32),
                                                       num_epochs=1,
                                                       shuffle=False)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None,
                                      throttle_secs=60, start_delay_secs=150)

    # Create the Estimator
    model_dir = '{}/{}'.format(MODELS_DIR, model_name)
    print('Saving model to {}'.format(model_dir))
    g_regression_model = tf.estimator.Estimator(model_fn=g_model_fn, params=params, model_dir=model_dir)

    # train and validate the estimator
    tf.estimator.train_and_evaluate(estimator=g_regression_model, train_spec=train_spec, eval_spec=eval_spec)

    # perform inference
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x=frame_to_dict(test_data),
                                                          num_epochs=1,
                                                          shuffle=False)
    predictions = g_regression_model.predict(input_fn=predict_input_fn)
    predictions = [p[0] for p in predictions]  # convert arrays to float

    # prepare submission
    submission = pd.DataFrame(predictions, columns=['y'])
    submission.index.name = 'Index'
    os.makedirs('./submissions/', exist_ok=True)
    submission.to_csv('./submissions/submission_{}.csv'.format(model_name))


def parameter_to_name(params):
    return 'LR{}_DR{}_HU{}_SH{}'.format(params['learning_rate'], params['dropout_rate'],
                                        '-'.join([str(x) for x in params['hidden_units']]),
                                        params['shuffle'])


def frame_to_dict(dataframe):
    input_dict = {
        'days': dataframe['Day'].values,
        'markets': dataframe['Market'].values,
        'stocks': dataframe['Stock'].values,
        'numerical_input': dataframe[NUMERICAL_COLS].values.astype(np.float32)
    }
    if 'Weight' in dataframe.columns:
        input_dict['weights'] = dataframe['Weight'].values.astype(np.float32)
    return input_dict


if __name__ == "__main__":
    tf.app.run(main)
