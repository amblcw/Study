import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())   # False, 즉시실행모드가 켜져있으면 True

# tf.compat.v1.disable_eager_execution()  # 즉시실행모드 끄기,  tf1에서 default
tf.compat.v1.enable_eager_execution()   # 즉시실행모드 켜기     tf2에서 default
print(tf.executing_eagerly())   # True

hellow = tf.constant('hellow world')
sess = tf.compat.v1.Session()
print(sess.run(hellow))

'''             실행가능여부
1.14.0  disable O
1.14.0  enable  X
2.9.0   disable O
2.9.0   enable  X
'''