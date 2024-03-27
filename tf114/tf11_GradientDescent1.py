import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(47)

x_train = [1,2,3]
y_train = [1,2,3]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)
w = tf.compat.v1.Variable([-2],dtype=tf.float32,name='weight')

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


with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("step      loss           weight")
    for step in range(21):
        _, loss_v, w_v = sess.run([update,loss_fn,w],feed_dict={x:x_train,y:y_train})
        w_hist.append(w_v[0])
        loss_hist.append(loss_v)
        print(f"{step:<10}{loss_v:<15.10f}{w_v[0]:<10.6f}")
        
print(loss_hist)
plt.subplot(121)
plt.plot(w_hist,color='red',label='weight',marker='.')
plt.legend()
plt.subplot(122) 
plt.plot(loss_hist,color='blue',label='loss',marker='.')
plt.legend()
plt.show()