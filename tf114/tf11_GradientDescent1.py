import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(47)

x_train = [1]
y_train = [1]

# x_train = [1,2,3,4,5]
# y_train = [3,5,7,9,11]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)
w = tf.compat.v1.Variable([0.1],dtype=tf.float32,name='weight')

################## model ##################
hypothesis = x*w

################## compile  // model.compile(loss='mse',optimizer='SGD')
loss_fn = tf.reduce_mean(tf.square(hypothesis - y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
lr = 0.1
gradient = tf.reduce_mean((x*w - y) * x)
descent = w - lr*gradient
update = w.assign(descent)

################## fit ##################
w_hist = []
loss_hist = []

pred = None
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("step      loss           weight")
    for step in range(5):
        _, loss_v, w_v = sess.run([update,loss_fn,w],feed_dict={x:x_train,y:y_train})
        w_hist.append(w_v[0])
        loss_hist.append(loss_v)
        print(f"{step:<10}{loss_v:<15.10f}{w_v[0]:<10.6f}")
        
    pred = sess.run(hypothesis,feed_dict={x:x_train,y:y_train})
    print(pred)
        
plt.subplot(131)
plt.plot(w_hist,color='red',label='weight',marker='.')
plt.legend()
plt.subplot(132) 
plt.plot(loss_hist,color='blue',label='loss',marker='.')
plt.legend()
plt.subplot(133)
plt.plot(x_train,y_train,color='blue',label='y_true',marker='.')
plt.plot(x_train,pred,color='red',label='y_pred',marker='.')
plt.legend()
plt.show()