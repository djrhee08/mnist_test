import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

num_points = 2000
vectors_set = []

for i in range(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.0,0.9),np.random.normal(0.0,0.9)])

    else:
        vectors_set.append([np.random.normal(3.0,0.5),np.random.normal(1.0,0.5)])

vectors = tf.constant(vectors_set)
k = 4
centroids = tf.Variables(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))

expanded_vectors = tf.expand_dims(vectors,0)
expanded_centroids = tf.expand_dims(centroids, 1)

assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors,expanded_centroids)), 2), 0)

means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignemnts, c)), [1,-1])),reduction_indices)])



df = pd.DataFrame({"x" : [v[0] for v in vectors_set],
                 "y" : [v[1] for v in vectors_set]})
sns.lmplot("x","y", data=df, fit_reg = False, size = 6)
plt.show()
