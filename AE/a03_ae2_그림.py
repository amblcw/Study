import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
import tensorflow as tf
import keras
tf.random.set_seed(47)
np.random.seed(47)

print(keras.__version__)

#data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float') / 255.
x_test = x_test.reshape(-1, 28*28).astype('float') / 255.
print(x_train.shape, x_test.shape) # (60000, 784) (10000, 784)

x_train_noised = x_train + np.random.normal(0,0.3, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0,0.3, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

print(np.max(x_train_noised), np.max(x_test_noised))
print(np.min(x_train_noised), np.min(x_test_noised))

# model
from keras.models import Sequential, Model
from keras.layers import Dense, Input

# def autoencoder(hidden_size:int):
#     input_img = Input(shape=(28*28))
#     encoded = Dense(hidden_size, activation='relu')(input_img)
#     decoded = Dense(28*28, activation='sigmoid')(encoded)
#     return Model(input_img,decoded)
def autoencoder(hidden_size:int):
    model = Sequential()
    model.add(Dense(28*28,input_shape=x_train_noised.shape[1:],activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(28*28,activation='sigmoid'))
    return model

model = autoencoder(64)
# model = autoencoder(154)
# model = autoencoder(331)
# model = autoencoder(486)
# model = autoencoder(713)
model.compile(optimizer='adam', loss='mse')
es = EarlyStopping(monitor='val_loss',mode='auto',patience=100,restore_best_weights=True)
model.fit(x_train_noised,x_train,epochs=10,batch_size=128,verbose=2,validation_split=0.2,callbacks=[es])

decoded_imgs = model.predict(x_test_noised)

n=10
plt.figure(figsize=(20,10))
# plt.gca().axes.yaxis.set_visible(False)
plt.axis('off')
for i in range(n):
    ax = plt.subplot(3,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.set_yticks([])
    if i==0:
        ax.set_ylabel("INPUT",size=20)

    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noised[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.set_yticks([])
    ax.grid(False)
    if i==0:
        ax.set_ylabel("NOISE",size=20)

    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.set_yticks([])
    if i==0:
        ax.set_ylabel("OUTPUT",size=20)

plt.show()