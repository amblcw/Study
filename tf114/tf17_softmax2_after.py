import tensorflow as tf
import numpy as np
tf.set_random_seed(47)

x_data = np.array([
    [1,2,1,1,],
    [2,1,3,2,],
    [3,1,3,4,],
    [4,1,5,5,],
    [1,7,5,5,],
    [1,2,5,6,],
    [1,6,6,6,],
    [1,7,6,7,],
    ])

row_y_data = [2,2,2,1,1,1,0,0,]
y_data = np.asarray(row_y_data).reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder
y_data = OneHotEncoder(sparse=False).fit_transform(y_data)
print(x_data.shape,y_data.shape)    # (8, 4) (8, 3)


x = tf.compat.v1.placeholder(tf.float32, shape=[None,4],name='x')
y = tf.compat.v1.placeholder(tf.float32, shape=[None,3],name='y')

w = tf.compat.v1.Variable(tf.random_normal([4,3]),shape=[4,3],name='weight')
b = tf.compat.v1.Variable(tf.zeros([1,3]),shape=[1,3],name='bias')

hypothesis = tf.nn.softmax(tf.add(tf.matmul(x,w),b))

loss_fn = tf.reduce_mean(-tf.reduce_mean(y*tf.log(hypothesis-y),axis=1)) # categorical_cross_entropy
loss_fn = tf.compat.v1.losses.softmax_cross_entropy(y_data,hypothesis)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

train = optimizer.minimize(loss_fn)

EPOCHS = 10000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1,EPOCHS+1):
        _, loss = sess.run([train,loss_fn],feed_dict={x:x_data,y:y_data})
        if step%1 == 0:
            print(f"{step}epo loss:{loss}")
        
    pred = sess.run(hypothesis,feed_dict={x:x_data,y:y_data})
    print(pred)
    argmax_pred = np.argmax(pred,axis=1)
    print(argmax_pred)
    
from sklearn.metrics import accuracy_score
acc = accuracy_score(argmax_pred,row_y_data)
print("ACC: ",acc)