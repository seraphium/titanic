# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
import numpy as np
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from graph import convert_data_to_tensors, get_network

#import for Random forest

from tensorflow.contrib.learn.python.learn\
        import metric_spec
from tensorflow.contrib.tensor_forest.client\
        import eval_metrics
from tensorflow.contrib.tensor_forest.client\
        import random_forest
from tensorflow.contrib.tensor_forest.python\
        import tensor_forest

tf.logging.set_verbosity(tf.logging.INFO)

import tempfile

from dataprocess import loaddata

train_df, _ = loaddata(normalization=True)

train_df = train_df.astype(np.float32)
x_train = train_df.drop('Survived', axis=1)
train_df['Unsurvived'] = 1 - train_df['Survived']
y_train = train_df.loc[:, ['Unsurvived', 'Survived']]

features = int(len(x_train.columns))
print('features:', features)
#
# log_reg = LogisticRegression()
# log_reg.fit(x_train, y_train)
#
# y_predict = log_reg.predict(x_test)
#
# acc_log = round(log_reg.score(x_train, y_train) * 100, 2)
#
# print('Logistic regression accuracy:', acc_log)
#
# # feature coefficients
# coeff_df = pd.DataFrame(train_df.columns.delete(0))
# coeff_df.columns = ['Feature']
# coeff_df["Correlation"] = pd.Series(log_reg.coef_[0])
#
# print(coeff_df.sort_values(by='Correlation', ascending=False))
#
#
# svc = SVC()
# svc.fit(x_train, y_train)
# y_predict = svc.predict(x_test)
# acc_svc = round(svc.score(x_train, y_train)*100, 2)
# print('SVM regression accuracy:', acc_svc)
#
# tf.logging.set_verbosity(tf.logging.INFO)


#low level tensorflow
#
# y_train = y_train.values.reshape(-1, 1)
# x_train = x_train.values
#
# b = tf.Variable(tf.zeros([1, 1]))
# x = tf.placeholder(tf.float32, [None, features])
#
# w = tf.Variable(tf.zeros([features, 1]))
#
# factor = tf.constant(1.0)
# #output
# z = tf.matmul(x, w) + b
# y = tf.nn.sigmoid(factor * z)
# y_ = tf.placeholder(tf.float32, [None, 1])
#
# lbda = tf.constant(0.00001)
#
# #cross entropy
# loss = -tf.reduce_mean(y_ * tf.log(y) + (1 - y_) * tf.log(1 - y)) + lbda * tf.nn.l2_loss(w)
#
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#
# correct_prediction = tf.less(tf.abs(y - y_), 0.5)
#
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#
#     epoch = 10000
#     for index in range(epoch):
#         _ = sess.run([train_step], feed_dict={x: x_train, y_: y_train})
#         if index % 100 == 0:
#             loss_value = sess.run([loss], feed_dict={x: x_train, y_: y_train})
#             print('loss:', loss_value)
#             acc_value = accuracy.eval(feed_dict={x: x_train, y_: y_train})
#             print('accuracy:', acc_value)
#
#     predict_value = list(map(lambda value : 1 if value > 0.5 else 0, list(y.eval(feed_dict={x: x_test}))))
#
#


COLUMNS = train_df.columns
FEATURES = COLUMNS[1:]
LABEL = COLUMNS[0]

'''
High level tf approach
'''
'''
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, hidden_units=[30, 30],
                                          model_dir="./titanic_model")

def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
    labels = tf.constant(data_set[LABEL].values)

    return feature_cols, labels

regressor.fit(input_fn=lambda: input_fn(train_df), steps=5000)

predict_value = regressor.predict(input_fn=lambda: input_fn(test_df))

binary_value = list(map(lambda value : 1 if value > 0.5 else 0, list(predict_value)))
'''


'''
random forest approach
'''
'''
params = tensor_forest.ForestHParams(
    num_classes=2, num_features=len(FEATURES),
    num_trees=100, max_nodes=1000)
graph_builder_class = tensor_forest.RandomForestGraphs

model_dir = tempfile.mkdtemp()

estimator = random_forest.TensorForestEstimator(
    params, graph_builder_class=graph_builder_class,
    model_dir=model_dir)

early_stopping_rounds = 100
monitor = random_forest.TensorForestLossHook(early_stopping_rounds)

estimator.fit(x=x_train.values, y=y_train.values,
              batch_size=1000, monitors=[monitor])
metric_name = 'accuracy'
metric = {metric_name:
    metric_spec.MetricSpec(
        eval_metrics.get_metric(metric_name),
        prediction_key=eval_metrics.get_prediction_key(metric_name))}

binary_value = estimator.predict(x=x_test.values, batch_size=1, as_iterable=False)
'''

''''
TF-Slim approach
'''


ckpt_path = "/tmp/titanic_train/"

with tf.Graph().as_default():

    inputs, targets = convert_data_to_tensors(x_train, y_train)
    predictions, nodes = get_network(inputs, is_training=True)

    mean_squared_error_loss = tf.losses.mean_squared_error(predictions, targets)

    tf.summary.scalar('loss/mean_squared_error_loss', mean_squared_error_loss)
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(targets , 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


    optimizer = tf.train.AdamOptimizer(learning_rate=0.03)
    train_op = slim.learning.create_train_op(mean_squared_error_loss, optimizer)
    logdir = ckpt_path

    final_loss = slim.learning.train(train_op, logdir,
                                     number_of_steps=100000,
                                     save_summaries_secs=10,
                                     log_every_n_steps=10000)
    print("Finished training.Final loss:", final_loss)

