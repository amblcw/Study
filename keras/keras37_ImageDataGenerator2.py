from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

train_data_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_data_gen = ImageDataGenerator(
    rescale=1./255
)

path = "C:\\_data\\etc\\brain\\"
path_train = path+"train\\"
path_test = path+"test\\"

xy_train = train_data_gen.flow_from_directory(
    path_train,
    target_size=(200,200),
    batch_size=160,
    class_mode='binary',
    shuffle=True
)

xy_test = test_data_gen.flow_from_directory(
    path_test,
    target_size=(200,200),
    batch_size=120,
    class_mode='binary',
    shuffle=False
)

x_train , y_train = (xy_train[0][0], xy_train[0][1])
print(f"{x_train.shape=}{y_train.shape=}")  #x_train.shape=(160, 200, 200, 3)y_train.shape=(160,)
x_test, y_test = (xy_test[0][0], xy_test[0][1])
print(f"{x_test.shape=}{y_test.shape=}")    #x_test.shape=(120, 200, 200, 3)y_test.shape=(120,)
hist = []

# model
model = Sequential()
model.add(Conv2D(32,(3,3),padding='valid',strides=2,input_shape=x_train.shape[1:]))
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3),padding='valid',strides=2))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Conv2D(32,(3,3),padding='valid',activation='relu'))
model.add(Conv2D(32,(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Conv2D(64,(2,2),padding='valid',activation='relu'))
model.add(Conv2D(64,(2,2),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# model = Sequential()
# model.add(Conv2D(32,(3,3),padding='same',strides=2,input_shape=x_train.shape[1:]))
# model.add(MaxPooling2D())
# model.add(Conv2D(32,(3,3),padding='same',strides=2))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
# model.add(Dropout(0.15))
# model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
# model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
# model.add(Dropout(0.15))
# model.add(Conv2D(64,(2,2),padding='valid',activation='relu'))
# model.add(Conv2D(64,(2,2),padding='valid',activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(1024,activation='relu'))
# model.add(Dense(512,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(128,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))

# compile & fit
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_loss',mode='auto',patience=200,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=1024,validation_data=(x_test,y_test),verbose=2,callbacks=[es])

# model = load_model(path+"model_save\\acc_0.9916666746139526.h5")

# evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)

print(f"LOSS: {loss[0]}\nACC:  {loss[1]}")
model.save(path+f"model_save\\acc_{loss[1]}.h5")

if hist != []:
    plt.title("brain CNN")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.plot(hist.history['val_acc'],label='val_acc',color='red')
    plt.plot(hist.history['acc'],label='acc',color='blue')
    # plt.plot(hist.history['val_loss'],label='val_loss',color='red')
    # plt.plot(hist.history['loss'],label='loss',color='blue')
    plt.legend()
    plt.show()

#acc > 0.99

# LOSS: 0.0754295364022255
# ACC:  0.9916666746139526