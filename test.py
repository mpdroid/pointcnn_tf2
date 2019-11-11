import tensorflow as tf
a = tf.constant([[[1,2]], [[3,4]]])
print(tf.shape(a))

b=tf.reshape(a,[2,2])
print(tf.shape(b))
print(b)
