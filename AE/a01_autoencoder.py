import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
import keras

print(keras.__version__)

#data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float') / 255.
x_test = x_test.reshape(-1, 28*28).astype('float') / 255.
print(x_train.shape, x_test.shape) # (60000, 784) (10000, 784)

from keras.models import Sequential, Model
from keras.layers import Dense, Input

input_img = Input(shape=(28*28))
encoded = Dense(64, activation='relu')(input_img)
decoded = Dense(28*28, activation='sigmoid')(encoded)

autoencoder = Model(input_img,decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
es = EarlyStopping(monitor='val_loss',mode='auto',patience=100,restore_best_weights=True)
autoencoder.fit(x_train,x_train,epochs=100,batch_size=128,verbose=2,validation_split=0.2,callbacks=[es])

decoded_imgs = autoencoder.predict(x_test)

n=10
plt.figure(figsize=(20,10))
for i in range(n):
    ax = plt.subplot(4,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(4, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

for i in range(n):
    ax = plt.subplot(4, n, i + 1 + 2*n)
    plt.imshow(x_test[i+n].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(4, n, i + 1 + 3*n)
    plt.imshow(decoded_imgs[i+n].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()