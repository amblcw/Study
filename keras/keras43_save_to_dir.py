from keras.datasets import fashion_mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, MaxPooling2D, Conv2D, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import time

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# plt.imshow(x_train[0], 'gray')
# plt.show()

print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")
# x_train.shape=(60000, 28, 28)
# x_test.shape=(10000, 28, 28)
# y_train.shape=(60000,)
# y_test.shape=(10000,)

# print(np.unique(y_test,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000], dtype=int64))

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255


aug_data_gen = ImageDataGenerator(
    # horizontal_flip=True,
    # vertical_flip=True,
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
    shuffle=False,
    save_to_dir="C:/_data/temp"
)

aug_x, aug_y = aug_data.next()  #한번 실행해야 저장된다

