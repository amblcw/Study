from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np
import tensorflow as tf
print(tf.__version__)   # 2.9.0
tf.random.set_seed(777) # 이쪽이 가중치 초기화에 영향
np.random.seed(777)

from keras.applications import VGG16
from keras.datasets import cifar10
from keras.callbacks import EarlyStopping
import time

# data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# model
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
vgg16.trainable = False # 가중치를 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()
# Total params: 14,914,378
# Trainable params: 199,690
# Non-trainable params: 14,714,688

# compile & fit
st = time.time()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss',mode='auto',patience=20,restore_best_weights=True)
model.fit(x_train, y_train, batch_size=512, validation_split=0.2, epochs=500, callbacks=[es])
et = time.time()

# eval & pred
loss = model.evaluate(x_test,y_test)
print(f"time: {et-st:.2f}sec \nloss: {loss[0]}\nacc:  {loss[1]}")

# 성능비교, 시간체크

# 원래 모델
# 32relu,64relu,drop0.1,64relu,128relu,drop0.1,Flatten,512
# time: 366.77252554893494sec
# LOSS: 2.1206843852996826
# ACC:  0.6802999973297119

# VGG16
# time: 200.33sec
# loss: 7.216104030609131
# acc:  0.583299994468689