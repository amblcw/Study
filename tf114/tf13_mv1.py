import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(47)

x1_data = [73., 93., 89., 96., 73.]
x2_data = [88., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 108., 70.]
y_data = [152., 185., 180., 196., 142.]

x1 = tf.compat.v1.placeholder(tf.float32, name='x1')
x2 = tf.compat.v1.placeholder(tf.float32, name='x2')
x3 = tf.compat.v1.placeholder(tf.float32, name='x3')
y = tf.compat.v1.placeholder(tf.float32, name='y')

w1 = tf.compat.v1.Variable(tf.random_normal([1]),dtype=tf.float32,name='weight1')
w2 = tf.compat.v1.Variable(tf.random_normal([1]),dtype=tf.float32,name='weight2')
w3 = tf.compat.v1.Variable(tf.random_normal([1]),dtype=tf.float32,name='weight3')
b = tf.compat.v1.Variable(0,dtype=tf.float32,name='bias')

hypothesis = x1*w1 + x2*w2 + x3*w3 + b

loss_fn = tf.reduce_mean(tf.square(hypothesis-y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001) 
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

train = optimizer.minimize(loss_fn)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_hist = []
    for step in range(1000):
        _, loss = sess.run([train,loss_fn],feed_dict={x1:x1_data,x2:x2_data,x3:x3_data,y:y_data})
        loss_hist.append(loss)
        if step % 10 == 9:
            print(f"{step+1}epo | loss={loss}")
    pred = sess.run(hypothesis,feed_dict={x1:x1_data,x2:x2_data,x3:x3_data,y:y_data})    
    
print(pred)
plt.subplot(1,2,1) 
plt.plot(loss_hist,color='blue',label='loss',marker='.')
plt.legend()
plt.subplot(1,2,2) 
plt.plot(y_data,color='blue',label='y_true',marker='.')
plt.plot(pred,color='red',label='y_pred',marker='.')
plt.legend()
plt.show()
    