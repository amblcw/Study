import tensorflow as tf
print(tf.__version__)   # 1.14.0
if tf.__version__ != '1.14.0':
    raise Exception("tf version is not 1.14.0")

# 3 + 4 = ?
node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
# node3 = node1 + node2
node3 = tf.add(node1,node2)
print(node3)    # Tensor("Add:0", shape=(), dtype=float32)

sess = tf.Session()
print(sess.run(node3))  # 7.0
