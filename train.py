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
import numpy as np

from dataprocess import loaddata

train_df, test_df = loaddata()

x_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']

x_test = test_df.drop('PassengerId', axis=1)

features = int(len(x_train.columns))
print('features:', features)

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

y_predict = log_reg.predict(x_test)

acc_log = round(log_reg.score(x_train, y_train) * 100, 2)

print('Logistic regression accuracy:', acc_log)

# feature coefficients
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(log_reg.coef_[0])

print(coeff_df.sort_values(by='Correlation', ascending=False))


svc = SVC()
svc.fit(x_train, y_train)
y_predict = svc.predict(x_test)
acc_svc = round(svc.score(x_train, y_train)*100, 2)
print('SVM regression accuracy:', acc_svc)


#tensorflow

y_train = y_train.values.reshape(-1, 1)
x_train = x_train.values

b = tf.Variable(tf.zeros([1, 1]))
x = tf.placeholder(tf.float32, [None, features])

w = tf.Variable(tf.zeros([features, 1]))

factor = tf.constant(1.0)
#output
z = tf.matmul(x, w) + b
y = tf.nn.sigmoid(factor * z)
y_ = tf.placeholder(tf.float32, [None, 1])

lbda = tf.constant(0.00001)

#cross entropy
loss = -tf.reduce_mean(y_ * tf.log(y) + (1 - y_) * tf.log(1 - y)) + lbda * tf.nn.l2_loss(w)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

correct_prediction = tf.less(tf.abs(y - y_), 0.5)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    epoch = 10000
    for index in range(epoch):
        _ = sess.run([train_step], feed_dict={x: x_train, y_: y_train})
        if index % 100 == 0:
            loss_value = sess.run([loss], feed_dict={x: x_train, y_: y_train})
            print('loss:', loss_value)
            acc_value = accuracy.eval(feed_dict={x: x_train, y_: y_train})
            print('accuracy:', acc_value)

    predict_value = list(map(lambda value : 1 if value > 0.5 else 0, list(y.eval(feed_dict={x: x_test}))))


#generate submission result
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": predict_value
    })

print(submission)

submission.to_csv("submission.csv", index=False)