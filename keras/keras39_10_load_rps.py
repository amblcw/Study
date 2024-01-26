from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np

IMAGE_SIZE = int(150)
path = "C:\\_data\\image\\rps\\"

# data
load_path = path+f"data_{IMAGE_SIZE}px"
x = np.load(load_path+"_x.npy")
y = np.load(load_path+"_y.npy")
print(f"{x.shape=} {y.shape=}") #x.shape=(2520, 150, 150, 3) y.shape=(2520, 3)

r = int(np.random.uniform(1,1000))
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r,stratify=y)

# model
model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=x_train.shape[1:], activation='relu'))
model.add(Conv2D(32,(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(2,2,),activation='relu'))
model.add(Conv2D(64,(2,2,),activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Conv2D(64,(2,2,),activation='relu'))
model.add(Conv2D(64,(2,2,),activation='relu'))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.summary()

# compile & fit
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_acc',mode='auto',patience=10,verbose=1)
hist = model.fit(x_train,y_train,epochs=1024,batch_size=128,validation_split=0.2,verbose=2,callbacks=[es])

# evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)

print(f"{r=}\nLOSS: {loss[0]}\nACC:  {loss[1]}")

# r=860
# LOSS: 0.0001959040528163314
# ACC:  1.0