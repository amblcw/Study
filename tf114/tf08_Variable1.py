import tensorflow as tf
tf.compat.v1.set_random_seed(777)

weights = tf.compat.v1.Variable(tf.random_normal([2]), name='weights')
print(weights)
# <tf.Variable 'weights:0' shape=(2,) dtype=float32, numpy=array([0.7706481 , 0.37335405], dtype=float32)>

# 초기화 첫번째
print("초기화 첫번째")
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    weights_result = sess.run(weights)
    print(weights_result)

# 초기화 두번째
print("초기화 두번째")
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    weights_result = weights.eval(session=sess)
    print(weights_result)
    
print("초기화 세번째")
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
weights_result = weights.eval()
print(weights_result)
sess.close()

