import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())
tf.compat.v1.disable_eager_execution()

# node1 = tf.constant(3.0)
# node2 = tf.constant(4.0)
# node3 = tf.add(node1,node2)

a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)
add_node = a + b

sess = tf.compat.v1.Session()

print(sess.run(add_node, feed_dict={a:3,b:4}))      # 7.0 꼭 노드 이름으로 해줘야함
print(sess.run(add_node, feed_dict={a:30,b:4.5}))   # 34.5

add_and_triple = add_node * 3
print(sess.run(add_and_triple, feed_dict={a:3,b:4}))# 21.0