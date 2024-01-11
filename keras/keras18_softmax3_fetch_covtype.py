from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

datasets = fetch_covtype()
x = datasets.data  
y = datasets.target

# print(x.shape,y.shape)      #(581012, 54) (581012,)
# print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747

# y = pd.get_dummies(y)
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y.reshape(-1,1))
y = to_categorical(y)

# print(y,y.shape,sep='\n')
# print(np.count_nonzero(y[:,0]))
'''
sklearn : (581012, 7)
pandas  : (581012, 7)
keras   : (581012, 8)
keras 첫번째 열이 미심직어 찍어보니
print(np.count_nonzero(y[:,0])) # 0
따라서 첫번째 열 잘라내고 슬라이싱
'''
y = y[:,1:]
# print(y,y.shape,sep='\n')       # (581012, 7)
# print(np.count_nonzero(y[:,0])) # 211840

r = int(np.random.uniform(1,1000))
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=r,stratify=y)

#model
model = Sequential()
model.add(Dense(128,input_dim=54,activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8,  activation='relu'))
model.add(Dense(7,  activation='softmax'))

#compile & fit
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_acc',mode='auto',patience=200,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=4096,batch_size=2048,validation_split=0.2, verbose=2,callbacks=[es])

#evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)
y_predict = np.argmax(model.predict(x_test),axis=1)
y_test = np.argmax(y_test,axis=1)
#만약 제출하게 된다면 1부터 시작하므로 모든 데이터에 +1을 해줘야함

print(f"{r=} \nLOSS: {loss[0]} \nACC:  {accuracy_score(y_test,y_predict)}({loss[1]} by loss[1])")

plt.title('Fetch covtype Classification')
plt.xlabel('epechs')
plt.ylabel('accuracy')
plt.plot(hist.history['acc'],label='acc',color='red')
plt.plot(hist.history['val_acc'],label='val_acc',color='blue')
plt.legend()
plt.show()

# r=696 
# LOSS: 0.24115876853466034
# ACC:  0.906577014870571(0.9065769910812378 by loss[1])