import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W)+b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

y_show = tf.argmax(y,1)


#print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

for i in range(100):
    temp = mnist.test.images[i]
    temp = temp.reshape(1,784)
    temp_label = mnist.test.labels[i]
    temp_label = temp_label.reshape(1,10)
    if not sess.run(correct_prediction, feed_dict={x: temp, y_ : temp_label}):
        print(sess.run(y_show, feed_dict={x: temp}), np.where(temp_label[0]==1)[0][0])
        # pixels = temp.reshape((28,28))
        # plt.imshow(pixels,cmap='gray')
        # plt.show()