from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

import warnings
warnings.filterwarnings(action='ignore')

#data
path = "C:\\_data\\KAGGLE\\bike-sharing-demand\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submission_csv = pd.read_csv(path+"sampleSubmission.csv")

x = train_csv.drop(['casual','registered','count'],axis=1)
y = train_csv['count']

from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

y = LabelEncoder().fit_transform(y)
print(x.shape,y.shape)
print(np.unique(y,return_counts=True))
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False)
x = pf.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8,
    # stratify=y
)

sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)


import tensorflow as tf
tf.compat.v1.random.set_random_seed(47)

x = tf.compat.v1.placeholder(tf.float32,shape=(None,x_train.shape[1]),name='x')
y = tf.compat.v1.placeholder(tf.float32,shape=(None,),name='y')

w = tf.compat.v1.Variable(tf.random_normal([x_train.shape[1],1]),shape=(x_train.shape[1],1),name='weight')
b = tf.compat.v1.Variable(0,dtype=tf.float32,name='bias')

hypothesis = tf.add(tf.matmul(x,w),b)

loss_fn = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

train = optimizer.minimize(loss_fn)

EPOCHS = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_hist = []
    for step in range(EPOCHS):
        _, loss = sess.run([train,loss_fn],feed_dict={x:x_train,y:y_train})
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

# r2: -0.204661013931128  |  mae: 135.97335885663554