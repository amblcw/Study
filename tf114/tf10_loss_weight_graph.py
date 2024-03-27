import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(47)

x = [1,2]
y = [1,2]

w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x*w

loss_fn = tf.reduce_mean(tf.square(hypothesis - y))

w_hist = []
loss_hist = []

with tf.compat.v1.Session() as sess:
    for i in range(-30,50):
        curr_w = i
        curr_loss = sess.run(loss_fn, feed_dict={w:curr_w})
        w_hist.append(curr_w)
        loss_hist.append(curr_loss)
        
    print(loss_hist)
    plt.subplot(121)
    plt.plot(w_hist,color='red',label='weight',marker='.')
    plt.subplot(122) 
    plt.plot(loss_hist,color='blue',label='loss',marker='.')
    plt.legend()
    plt.show()