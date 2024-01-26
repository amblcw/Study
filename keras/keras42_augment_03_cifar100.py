from keras.datasets import cifar100
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import time
import matplotlib.pylab as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from function_package import image_scaler
from keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

xtr0, xtr1, xtr2, xtr3 = (x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])
xt0, xt1, xt2, xt3 = (x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])

# x_train = x_train.reshape(xtr0, xtr1*xtr2*xtr3)
# x_test = x_test.reshape(xt0, xt1*xt2*xt3)

# minmax = MinMaxScaler().fit(x_train)
# x_train = minmax.transform(x_train)
# x_test = minmax.transform(x_test)

# standard = StandardScaler().fit(x_train)
# x_train = standard.transform(x_train)
# x_test = standard.transform(x_test)

# x_train = x_train.reshape(xtr0, xtr1, xtr2, xtr3)
# x_test = x_test.reshape(xt0, xt1, xt2, xt3)

# x_train = x_train.astype(np.float32) - 127.5
# x_test = x_test.astype(np.float32) - 127.5
# x_train /= 127.5
# x_test /= 127.5

x_train, x_test = image_scaler(x_train,x_test,'robust')

print(np.min(x_train),np.max(x_train))


aug_data_gen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    shear_range=30,
    zoom_range=0.2,
    fill_mode='nearest',
)

aug_data = aug_data_gen.flow(
    x_train,
    y_train,
    batch_size=x_train.shape[0],
    shuffle=False
)

aug_x, aug_y = aug_data.next()

print(f"{aug_x.shape=} {aug_y.shape=}")

x_train = np.concatenate([x_train,aug_x])
y_train = np.concatenate([y_train,aug_y])

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")


# model
# model = Sequential()
# model.add(Conv2D(32, (2,2), input_shape=x_train.shape[1:], activation='relu'))
# model.add(Conv2D(32, (2,2),activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(32, (2,2),activation='relu'))
# model.add(Conv2D(32, (2,2),activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(32, (2,2),activation='relu'))
# model.add(Conv2D(32, (2,2),activation='relu'))
# model.add(BatchNormalization())
# # model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(100, activation='softmax'))

input = Input(shape=x_train.shape[1:])
c1 = Conv2D(32,(2,2),activation='relu')(input)
c2 = Conv2D(32,(2,2),activation='relu')(c1)
b1 = BatchNormalization()(c2)
m1 = MaxPooling2D()(b1)
dr1 = Dropout(0.25)(m1)
c3 = Conv2D(32,(2,2),activation='relu')(dr1)
c4 = Conv2D(32,(2,2),activation='relu')(c3)
b2 = BatchNormalization()(c4)
m2 = MaxPooling2D()(b2)
dr2 = Dropout(0.25)(m2)
c5 = Conv2D(32,(2,2),activation='relu')(dr2)
c6 = Conv2D(32,(2,2),activation='relu')(c5)
b3 = BatchNormalization()(c6)
dr3 = Dropout(0.25)(b3)
f1 = Flatten()(dr3)
d1 = Dense(1024, activation='relu')(f1)
dr4 = Dropout(0.2)(d1)
output = Dense(100, activation='softmax')(dr4)

model = Model(inputs=input,outputs=output)




# compile & fit
start_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc', mode='max', patience=50,verbose=1)
hist = model.fit(x_train, y_train, epochs=1024, batch_size=2048, validation_data=(x_test,y_test), verbose=2, callbacks=es)
end_time = time.time()

# evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)

print(f"time: {end_time-start_time}sec")
print(f"LOSS: {loss[0]}\nACC:  {loss[1]}")

model.save(f"C:/_data/_save/keras31/CNN6_cifar100_ACC{loss[1]}.h5")

plt.title("CIFAR100")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.plot(hist.history['acc'],label='acc',color='red')
plt.plot(hist.history['val_acc'],label='val_acc',color='blue')
plt.legend()
plt.show()

# 0.4 넘기기

# time: 573.5564851760864sec
# LOSS: 2.015946626663208
# ACC:  0.5001000165939331

# time: 280.78862738609314sec
# LOSS: 10.030982971191406
# ACC:  0.26170000433921814

# function
# time: 182.88679814338684sec
# LOSS: 2.476529836654663
# ACC:  0.4555000066757202

# scaled MinMaxScaler
# time: 173.21599221229553sec
# LOSS: 2.494518518447876
# ACC:  0.4523000121116638

# scaled StandardScaler
# time: 183.86233830451965sec
# LOSS: 2.5694944858551025
# ACC:  0.45649999380111694

# scaled Robust
# time: 149.1822190284729sec
# LOSS: 2.403017044067383
# ACC:  0.46970000863075256

# augment
# time: 586.932817697525sec
# LOSS: 2.194991111755371
# ACC:  0.4645000100135803