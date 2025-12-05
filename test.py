import tensorflow as tf

with tf.device('/GPU:0'):
    a = tf.random.uniform((2000, 2000))
    b = tf.random.uniform((2000, 2000))
    c = tf.matmul(a, b)

print("Done")
