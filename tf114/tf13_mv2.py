import tensorflow as tf
tf.compat.v1.random.set_random_seed(47)

x_data = [[73,51,65,],
          [92,98,11,],
          [89,31,33,],
          [99,33,100],
          [17,66,79,],
          ]
y_data = [[152],[185],[180],[205],[142]]

x = tf.compat.v1.placeholder(tf.float32,shape=(5,3),name='x')
y = tf.compat.v1.placeholder(tf.float32,shape=(5,1),name='y')

w = tf.compat.v1.Variable(tf.random_normal([3,1]),shape=(3,1),name='weight')
b = tf.compat.v1.Variable(0,dtype=tf.float32,name='bias')

hypothesis = tf.add(tf.matmul(x,w),b)

loss_fn = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

train = optimizer.minimize(loss_fn)

EPOCHS = 10000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_hist = []
    for step in range(EPOCHS):
        _, loss = sess.run([train,loss_fn],feed_dict={x:x_data,y:y_data})
        if (step+1)%10 == 0:
            print(f"{step+1}epo | loss={loss}")
        loss_hist.append(loss)
    pred = sess.run(hypothesis,feed_dict={x:x_data,y:y_data})
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_data,pred)
    mae = mean_absolute_error(y_data,pred)
    print(f"r2: {r2}  |  mae: {mae}")
        
import matplotlib.pyplot as plt
print(pred)
plt.subplot(1,2,1) 
plt.plot(loss_hist,color='blue',label='loss',marker='.')
plt.legend()
plt.subplot(1,2,2) 
plt.plot(y_data,color='blue',label='y_true',marker='.')
plt.plot(pred,color='red',label='y_pred',marker='.')
plt.legend()
plt.show()