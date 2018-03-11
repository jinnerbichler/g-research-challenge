import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
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

    for lr in [1e-3, 1e-4, 1e-5, 1e-6]:
        for hu in hidden_units:
            for shuffle in [True, False]:
                try:
                    evaluate(learning_rate=lr, dropout_rate=0.0, hidden_units=hu, shuffle=shuffle,
                             batch_size=100, l2_reg_scale=0.0)
                except Exception as ex:
                    print(ex)


def evaluate(learning_rate, dropout_rate, hidden_units, shuffle, batch_size, l2_reg_scale, max_steps=7e5):
    train_data = pd.read_csv('data/train.csv', index_col=0).fillna(method='ffill')
    test_data = pd.read_csv('data/test.csv', index_col=0).fillna(method='ffill')
    all_data = pd.concat([train_data, test_data])

    # normalize continous data
    normalize_data(train=train_data, test=test_data)

    train_data, eval_data = train_test_split(train_data, test_size=0.40, shuffle=shuffle)
    # eval_data = train_data[train_data['Stock'] == 467]
    # train_data = train_data[train_data['Stock'] == 1]

    # create hyperparameter
    params = {
        'num_days': all_data['Day'].max(),
        'num_markets': all_data['Market'].max(),
        'num_stocks': all_data['Stock'].max(),
        'reg_scale': l2_reg_scale,
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'hidden_units': hidden_units,
        'shuffle': shuffle,
        'batch_size': batch_size
    }

    model_name = parameter_to_name(params)
    print('Start training with model {}'.format(model_name))

    # specify training
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x=make_features(train_data),
                                                        y=make_target(train_data),
                                                        batch_size=batch_size,
                                                        num_epochs=1,
                                                        shuffle=shuffle)
    max_steps = int(7e5)
    # hooks = [tf_debug.TensorBoardDebugHook("tensorboard:8080")]
    hooks = []
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=max_steps, hooks=hooks)

    # specify validation
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x=make_features(eval_data),
                                                       y=make_target(eval_data),
                                                       num_epochs=1,
                                                       shuffle=False)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None, hooks=hooks,
                                      throttle_secs=180, start_delay_secs=180)

    # Create the Estimator
    model_dir = '{}/{}'.format(MODELS_DIR, model_name)
    print('Saving model to {}'.format(model_dir))
    g_regression_model = tf.estimator.Estimator(model_fn=g_model_fn, params=params,
                                                model_dir=model_dir)

    # train and validate the estimator
    tf.estimator.train_and_evaluate(estimator=g_regression_model, train_spec=train_spec,
                                    eval_spec=eval_spec)

    # # prepare submission
    # predict_input_fn = tf.estimator.inputs.numpy_input_fn(x=make_features(test_data),
    #                                                       num_epochs=1,
    #                                                       shuffle=False)
    # predictions = g_regression_model.predict(input_fn=predict_input_fn)
    # predictions = [p['y'][0] for p in predictions]  # convert arrays to float
    #
    # submission = pd.DataFrame(predictions, columns=['y'])
    # submission.index.name = 'Index'
    # os.makedirs('./submissions/', exist_ok=True)
    # submission.to_csv('./submissions/submission_{}.csv'.format(model_name))

    # compute last validation
    eval_predictions = g_regression_model.evaluate(input_fn=eval_input_fn, steps=None)
    return eval_predictions['loss']


def normalize_data(train, test, min_max=True):
    all_data = pd.concat([train, test])

    if min_max:
        # scale data between -1.0 and 1.0
        max_vals = all_data[NUMERICAL_COLS].max()
        min_vals = all_data[NUMERICAL_COLS].min()
        train[NUMERICAL_COLS] = 2 * (train[NUMERICAL_COLS] - min_vals) / (max_vals - min_vals) - 1
        test[NUMERICAL_COLS] = 2 * (test[NUMERICAL_COLS] - min_vals) / (max_vals - min_vals) - 1

    else:
        # remove mean and adapt stddev of numerical columns
        numerical_mean = all_data[NUMERICAL_COLS].mean()
        numerical_stdev = all_data[NUMERICAL_COLS].std()
        train[NUMERICAL_COLS] = (train[NUMERICAL_COLS] - numerical_mean) / numerical_stdev
        test[NUMERICAL_COLS] = (test[NUMERICAL_COLS] - numerical_mean) / numerical_stdev


def g_model_fn(features, labels, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.name_scope('input'):
        days_one_hot = tf.one_hot(features['days'], depth=params['num_days'], name='day_one_hot')
        markets_one_hot = tf.one_hot(features['markets'], depth=params['num_markets'], name='market_one_hot')
        stocks_one_hot = tf.one_hot(features['stocks'], depth=params['num_stocks'], name='stocks_one_hot')

        merged = [features['numerical_input'], days_one_hot, markets_one_hot, stocks_one_hot]
        net = tf.concat(values=merged, axis=1, name='merged_input')

    with tf.name_scope('net'):
        for counter, hidden_unit in enumerate(params['hidden_units']):
            hu_name = 'fc_{}'.format(counter)
            with tf.name_scope(hu_name):
                regularizer = tf.contrib.layers.l2_regularizer(scale=params['reg_scale'])
                net = tf.layers.dense(inputs=net, units=hidden_unit,
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                      activation=tf.nn.tanh, kernel_regularizer=regularizer, name=hu_name)

                dense_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, hu_name)
                tf.summary.histogram('kernel', dense_vars[0], family=hu_name)
                tf.summary.histogram('bias', dense_vars[1], family=hu_name)
                tf.summary.histogram('activation', net, family=hu_name)

                net = tf.layers.dropout(inputs=net, rate=params['dropout_rate'], training=is_training)

        # predict y
        output_name = 'fc_out'
        with tf.name_scope(output_name):
            regularizer = tf.contrib.layers.l2_regularizer(scale=params['reg_scale'])
            predictions = tf.layers.dense(inputs=net, units=1, name=output_name,
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                          kernel_regularizer=regularizer)

            dense_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, output_name)
            tf.summary.histogram('kernel', dense_vars[0], family=output_name)
            tf.summary.histogram('bias', dense_vars[1], family=output_name)
            tf.summary.histogram('activation', predictions, family=output_name)

    # check if inference
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'y': predictions}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    with tf.name_scope('loss'):
        # weighted MSE
        wmse_loss = tf.reduce_mean(tf.pow(labels['y'] - predictions, 2) * labels['weight'],
                                   name='wmse_loss')

        # l2 regularizer
        l2_loss = tf.losses.get_regularization_loss()

        # total loss
        loss = tf.add(wmse_loss, l2_loss, name='total_loss')

    tf.summary.scalar('wmse_loss', wmse_loss, family='losses')
    tf.summary.scalar('l2_loss', l2_loss, family='losses')
    tf.summary.scalar('total_loss', loss, family='losses')

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    return tf.estimator.EstimatorSpec(mode=mode, loss=wmse_loss)


def parameter_to_name(params):
    return 'LR={},DR={:.1f},HU={},SH={},RS={:.5f},BS={}'.format(
        params['learning_rate'], params['dropout_rate'], '-'.join([str(x) for x in params['hidden_units']]),
        params['shuffle'], params['reg_scale'], params['batch_size'])


def make_features(dataframe):
    return {
        'days': dataframe['Day'].values,
        'markets': dataframe['Market'].values,
        'stocks': dataframe['Stock'].values,
        'numerical_input': dataframe[NUMERICAL_COLS].values.astype(np.float32)
    }


def make_target(dataframe):
    return {
        'y': dataframe['y'].values.astype(np.float32),
        'weight': dataframe['Weight'].values.astype(np.float32)
    }


if __name__ == "__main__":
    tf.app.run(main)
