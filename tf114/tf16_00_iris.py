from sklearn.datasets import load_iris
import tensorflow as tf
RANDOM_STATE = 47
tf.set_random_seed(RANDOM_STATE)
# data
x, y = load_iris(return_X_y=True)

x = x[y != 2]
y = y[y != 2]
y = y.reshape(-1,1)

print(y.shape) # (100,1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=RANDOM_STATE,stratify=y)

x = tf.compat.v1.placeholder(tf.float32,shape=[None,4])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w = tf.compat.v1.Variable(tf.random_normal([4,1]),dtype=tf.float32,shape=(4,1))
b = tf.compat.v1.Variable(0,dtype=tf.float32)

hypothesis = tf.sigmoid(tf.add(tf.matmul(x,w),b))

loss_fn = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))
# optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)

train = optimizer.minimize(loss_fn)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    EPOCHS = 1000
    for step in range(1,EPOCHS+1):
        _, loss, w_v, b_v = sess.run([train,loss_fn, w, b],feed_dict={x:x_train,y:y_train})
        if step % 10 == 0:
            # print(f"{step}epo loss={loss} weight=[{w_v}] bias={b_v}")
            print(f"{step}epo loss={loss}")
            
    pred, final_loss = sess.run([hypothesis,loss_fn],feed_dict={x:x_test,y:y_test})
    
import numpy as np
# print(pred)
pred = np.around(pred)
print("rounded pred",pred)
print("final loss:", final_loss)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,pred)
print("ACC: ",acc)

# final loss: 0.20886293
# ACC:  1.0