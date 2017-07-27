

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pandas as pd
from graph import convert_data_to_tensors, get_network

tf.logging.set_verbosity(tf.logging.INFO)

from dataprocess import loaddata

_, test_df = loaddata(normalization=True)

test_df = test_df.astype(np.float32)
x_test = test_df.drop('PassengerId', axis=1)


ckpt_path = "/tmp/titanic_train/"


with tf.Graph().as_default():
   # saver = tf.train.import_meta_graph("/tmp/titanic_train/model.ckpt-1000.meta")
    #with tf.Session() as sess:
    #    saver.restore(sess, "/tmp/titanic_train/model.ckpt-1000")

    inputs, _ = convert_data_to_tensors(x_test)
    predictions, end_points = get_network(inputs, is_training=False)

    sv = tf.train.Supervisor(logdir=ckpt_path)
    with sv.managed_session() as sess:
        inputs, predictions = sess.run([inputs, predictions])
        predictArgMax = np.argmax(predictions, 1)
        print(predictArgMax)


#generate submission result
submission = pd.DataFrame({
       "PassengerId": test_df["PassengerId"].astype(int),
       "Survived": predictArgMax
   })

print(submission)

submission.to_csv("submission.csv", index=False)
