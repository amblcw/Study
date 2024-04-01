from sklearn.datasets import load_breast_cancer
import tensorflow as tf
import numpy as np
RANDOM_STATE = 47
tf.set_random_seed(RANDOM_STATE)
# data
x, y = load_breast_cancer(return_X_y=True)

print(x.shape, y.shape) # (569, 30) (569,)
y = y.reshape(-1,1)
print(np.unique(y,return_counts=True))  # (array([0, 1]), array([212, 357], dtype=int64))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=RANDOM_STATE,stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x = tf.compat.v1.placeholder(tf.float32,shape=[None,30])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w = tf.compat.v1.Variable(tf.random_normal([30,1]),dtype=tf.float32,shape=(30,1))
b = tf.compat.v1.Variable(0,dtype=tf.float32)

hypothesis = tf.sigmoid(tf.add(tf.matmul(x,w),b))

loss_fn = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)

train = optimizer.minimize(loss_fn)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    EPOCHS = 1000
    for step in range(1,EPOCHS+1):
        _, loss, w_v, b_v = sess.run([train,loss_fn, w, b],feed_dict={x:x_train,y:y_train})
        if step % 1 == 0:
            # print(f"{step}epo loss={loss} weight=[{w_v}] bias={b_v}")
            print(f"{step}epo loss={loss}")
            
    pred, final_loss = sess.run([hypothesis,loss_fn],feed_dict={x:x_test,y:y_test})
    
pred = np.around(pred)
print("rounded pred",pred)
print("final loss:", final_loss)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,pred)
print("ACC: ",acc)
