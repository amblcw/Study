import tensorflow as tf
print(tf.__version__)   # 1.14.0
if tf.__version__ != '1.14.0':
    raise Exception("tf version is not 1.14.0")

hello = tf.constant('hello world')
print(hello)    # Tensor("Const:0", shape=(), dtype=string)

# 텐서 머신을 session 이라고 부름
sess = tf.Session()     # session 정의
print(sess.run(hello))  # 출력은 sess.run을 통해 출력함