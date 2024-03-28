import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(47)

x_train = [1,2,3,4,5]
y_train = [3,5,7,9,11]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)
w = tf.compat.v1.Variable(tf.random.normal([1]),dtype=tf.float32,name='weight')
b = tf.compat.v1.Variable(0,dtype=tf.float32,name='bias')

################## model ##################
hypothesis = x*w + b

################## compile  // model.compile(loss='mse',optimizer='SGD')
loss_fn = tf.reduce_mean(tf.square(hypothesis - y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
lr = 0.1
gradient = tf.reduce_mean((x*w + b - y) * x)
descent = w - lr*gradient
bias_gradient = tf.reduce_mean(x*w + b - y)
bias_decent = b - lr*bias_gradient
update = w.assign(descent)
bias_update = b.assign(bias_decent)

################## fit ##################
w_hist = []
b_hist = []
loss_hist = []


with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("step      loss           weight")
    for step in range(500):
        _, _, loss_v, w_v, b_v = sess.run([update,bias_update,loss_fn,w,b],feed_dict={x:x_train,y:y_train})
        w_hist.append(w_v[0])
        b_hist.append(b_v)
        loss_hist.append(loss_v)
        print(f"{step:<10}{loss_v:<15.10f}{w_v[0]:<10.6f}{b_v:<10.6f}")
        
    pred = sess.run(hypothesis,feed_dict={x:x_train,y:y_train})
    print(pred)
        
# print(loss_hist)
plt.subplot(2,2,1)
plt.plot(w_hist,color='red',label='weight',marker='.')
plt.legend()
plt.subplot(2,2,2) 
plt.plot(b_hist,color='green',label='bias',marker='.')
plt.legend()
plt.subplot(2,2,3) 
plt.plot(loss_hist,color='blue',label='loss',marker='.')
plt.legend()
plt.subplot(2,2,4) 
plt.plot(x_train,y_train,color='blue',label='y_true',marker='.')
plt.plot(x_train,pred,color='red',label='y_pred',marker='.')
plt.legend()
plt.show()