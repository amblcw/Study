import tensorflow as tf
tf.compat.v1.set_random_seed(47)

#1 data
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]  # (6,2)
y_data = [[0],[0],[0],[1],[1],[1]]              # (6,1)

#2 model
x = tf.compat.v1.placeholder(tf.float32,shape=[None,2],name='x')
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1],name='y')

w = tf.compat.v1.Variable(tf.random_normal([2,1]),dtype=tf.float32,name='weight')
b = tf.compat.v1.Variable(0,dtype=tf.float32,name='bias')

hypothesis = tf.compat.v1.sigmoid(tf.add(tf.matmul(x,w),b))

#3 compile 
# loss_fn = tf.losses.sigmoid_cross_entropy()
loss_fn = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

train = optimizer.minimize(loss_fn)

#4 train
EPOCHS = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1,EPOCHS+1):
        _, loss, w_v, b_v = sess.run([train,loss_fn, w, b],feed_dict={x:x_data,y:y_data})
        if step % 10 == 0:
            print(f"{step}epo loss={loss} weight=[{w_v[0]} {w_v[1]}] bias={b_v}")
            
    pred, final_loss = sess.run([hypothesis,loss_fn],feed_dict={x:x_data,y:y_data})
    
import numpy as np
print(pred)
pred = np.around(pred)
print("rounded pred",pred)
print("final loss:", final_loss)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_data,pred)
print("ACC: ",acc)
