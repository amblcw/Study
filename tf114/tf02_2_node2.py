import tensorflow as tf

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

# 실습
node3 = tf.add(node1,node2)
node4 = tf.subtract(node1,node2)
node5 = tf.multiply(node1,node2)
node6 = tf.divide(node1,node2)

sess = tf.Session()

print(sess.run(node3))  # 5.0
print(sess.run(node4))  # -1.0
print(sess.run(node5))  # 6.0
print(sess.run(node6))  # 0.6666667