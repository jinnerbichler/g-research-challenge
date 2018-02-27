import pandas as pd
import tensorflow as tf

train_data = pd.read_csv('data/train.csv', index_col=0)
test_data = pd.read_csv('data/test.csv', index_col=0)
NUMERICAL_COLS = ['x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6']
all_data = pd.concat([train_data, test_data])


# normalize numerical data
def normalize(train, test):
    all = pd.concat([train, test])
    normalized_data = (all - all.min()) / (all.max() - all.min())
    normalized_train = normalized_data.iloc[0:len(train)]
    normalized_test = normalized_data.iloc[len(train):]

    train[NUMERICAL_COLS] = normalized_train[NUMERICAL_COLS]
    test[NUMERICAL_COLS] = normalized_test[NUMERICAL_COLS]


# normalize continous data
normalize(train=train_data, test=test_data)

# create one-hot encoding
num_days = all_data['Day'].max()
num_markets = all_data['Market'].max()
num_stocks = all_data['Stock'].max()

with tf.name_scope('input'):
    day_ph = tf.placeholder(tf.int32, shape=[None], name='day')
    market_ph = tf.placeholder(tf.int32, shape=[None], name='market')
    stock_ph = tf.placeholder(tf.int32, shape=[None], name='stock')
    numerical_ph = tf.placeholder(tf.float32, shape=(None, len(NUMERICAL_COLS)), name='numeric_input')
    train_y_ph = tf.placeholder(tf.float32, shape=[None], name='training_y')
    train_w_ph = tf.placeholder(tf.float32, shape=[None], name='training_w')

    days_one_hot = tf.one_hot(day_ph, depth=num_days)
    markets_one_hot = tf.one_hot(market_ph, depth=num_markets)
    stocks_one_hot = tf.one_hot(stock_ph, depth=num_stocks)

    all_inputs = tf.concat([days_one_hot, markets_one_hot, stocks_one_hot, numerical_ph], 1)

with tf.name_scope('net'):
    dense = tf.layers.dense(inputs=all_inputs, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)
    predicted_y = tf.layers.dense(inputs=dropout, units=1)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.pow(train_y_ph - predicted_y, 2) * train_w_ph, name='loss')  # weighted MSE

with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for epoch in range(len(train_data)):
        epoch_data = train_data.ix[epoch]
        day = [epoch_data['Day']]
        market = [epoch_data['Market']]
        stock = [epoch_data['Stock']]
        numerical_data = [epoch_data[NUMERICAL_COLS].values]
        train_y = [epoch_data['y']]
        train_w = [epoch_data['Weight']]

        _, _, loss_r = sess.run([train_op, days_one_hot, loss], feed_dict={day_ph: day,
                                                                market_ph: market,
                                                                stock_ph: stock,
                                                                numerical_ph: numerical_data,
                                                                train_y_ph: train_y,
                                                                train_w_ph: train_w})

        print(loss_r)

        # # Display logs per epoch step
        # if (epoch+1) % display_step == 0:
        #     c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
        #     print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
        #         "W=", sess.run(W), "b=", sess.run(b))
