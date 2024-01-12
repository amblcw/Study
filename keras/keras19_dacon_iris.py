from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import sklearn.preprocessing

#data
path = "C:\\_data\\DACON\\iris\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submit_csv = pd.read_csv(path+"sample_submission.csv")

x = train_csv.drop(['species'],axis=1)
y = train_csv['species']

print(x,y,sep='\n')
print(x.shape,y.shape)  #x:(120, 4) y:(120,) test_csv:(30, 4)

y = y.to_frame('species')                                      #pandas Series에서 dataframe으로 바꾸는 법
ohe = sklearn.preprocessing.OneHotEncoder(sparse=False)     
y = ohe.fit_transform(y)

# print(x)
print(f"{x.shape=}, {y.shape=}")      
# print(np.unique(y, return_counts=True)) 

r = int(np.random.uniform(1,1000))
# r=326
x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r,stratify=y)

#model
model = Sequential()
model.add(Dense(128,input_dim=4,activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8,  activation='relu'))
model.add(Dense(3,  activation='softmax'))

#compile & fit
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_acc',mode='max',patience=30,restore_best_weights=True,verbose=1)
print(x_train.shape,y_train.shape)
hist = model.fit(x_train,y_train,epochs=1024,batch_size=1,validation_split=0.4,verbose=2,callbacks=[es])

#evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)
y_test = np.argmax(y_test,axis=1)                   #둘다 (n,)의 백터형태로 출력되므로 accuracy_score가능
y_predict = np.argmax(model.predict(x_test),axis=1)
y_submit = np.argmax(model.predict(test_csv),axis=1)

#결과값 출력
print(f"{r=}\nLOSS: {loss[0]}\nACC:  {accuracy_score(y_test,y_predict)}({loss[1]}by loss[1])")

#테스트 잘 분리되었는지 확인 및 예측결과값 비교
# print(np.unique(y_test,return_counts=True))
# print(np.unique(y_predict,return_counts=True))

#y_submit
import datetime
dt = datetime.datetime.now()
submit_csv['species'] = y_submit
submit_csv.to_csv(path+f"iris_{dt.day}day{dt.hour:2}{dt.minute:2}_ACC{loss[1]:5}.csv",index=False)

#그래프 출력
plt.figure(figsize=(12,9))
plt.title("DACON Iris Classification")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.plot(hist.history['acc'],label='acc',color='red')
plt.plot(hist.history['val_acc'],label='val_acc',color='blue')
plt.legend()
# plt.show()


# r=326
# LOSS: 0.3992086946964264
# ACC:  1.0(1.0by loss[1])