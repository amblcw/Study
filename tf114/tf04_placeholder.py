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
c = tf.compat.v1.placeholder(tf.int32)
d = tf.compat.v1.placeholder(tf.int32)
e = tf.compat.v1.placeholder(tf.float32)

add_node2 = c + d
# add_node3 = c + e

sess = tf.compat.v1.Session()

print(sess.run(add_node, feed_dict={a:3,b:4}))      # 7.0 꼭 노드 이름으로 해줘야함
print(sess.run(add_node, feed_dict={a:30,b:4.5}))   # 34.5
print(sess.run(add_node2, feed_dict={c:3.5,d:4.1})) # 7 소수점 무시됨
# print(sess.run(add_node3, feed_dict={c:3.5,e:4.1})) 
# TypeError: Input 'y' of 'AddV2' Op has type float32 that does not match type int32 of argument 'x'.
# int와 float를 서로 더할 수 없다
add_and_triple = add_node * 3
print(sess.run(add_and_triple, feed_dict={a:3,b:4}))# 21.0