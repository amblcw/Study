from sklearn.datasets import load_diabetes
import tensorflow as tf
tf.compat.v1.random.set_random_seed(47)

x,y = load_diabetes(return_X_y=True)
from sklearn.model_selection import train_test_split

x_data, x_test, y_data, y_test = train_test_split(x,y,train_size=0.8,random_state=47)
print(x_data.shape, x_test.shape, y_data.shape, y_test.shape)
# (353, 10) (89, 10) (353,) (89,)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler().fit(x_data)
x_data = scaler.transform(x_data)
x_test = scaler.transform(x_test)

x = tf.compat.v1.placeholder(tf.float32,shape=(None,10),name='x')
y = tf.compat.v1.placeholder(tf.float32,shape=(None,),name='y')

w = tf.compat.v1.Variable(tf.random_normal([10,1]),shape=(10,1),name='weight')
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
    pred = sess.run(hypothesis,feed_dict={x:x_test,y:y_test})
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_test,pred)
    mae = mean_absolute_error(y_test,pred)
    print(f"r2: {r2}  |  mae: {mae}")
        
import matplotlib.pyplot as plt
# print(pred)
plt.subplot(1,2,1) 
plt.plot(loss_hist,color='blue',label='loss',marker='.')
plt.legend()
plt.subplot(1,2,2) 
plt.plot(y_test,color='blue',label='y_true',marker='.')
plt.plot(pred,color='red',label='y_pred',marker='.')
plt.legend()
plt.show()

# r2: 0.07942899529580372  |  mae: 59.629804804084