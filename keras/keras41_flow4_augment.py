from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(f"{x_train.shape=}, {y_train.shape=}, {x_test.shape=}, {y_test.shape=}")
# x_train.shape=(60000, 28, 28), y_train.shape=(60000,), x_test.shape=(10000, 28, 28), y_test.shape=(10000,)
x_train = x_train.reshape(-1,28,28,1) / 255.
x_test = x_test.reshape(-1,28,28,1) / 255.

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    shear_range=30,
    zoom_range=0.2,
    fill_mode='nearest',
    # validation_split=0.2
)

augment_size = 40000

randidx = np.random.randint(x_train.shape[0], size=augment_size)    # 6만개중 4만개 뽑기
# print(np.min(randidx), np.max(randidx)) # 0 59997

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
# print(x_augmented.shape, y_augmented.shape)    # (40000, 28, 28) (40000,)

augmented_data = train_datagen.flow(
    x_augmented.reshape(-1,28,28,1),    # x
    y_augmented,                                                 # y
    batch_size=augment_size,
    shuffle=False
)

aug_x, aug_y = augmented_data.next()
print(f"{aug_x.shape=} {aug_y.shape=}") # aug_x.shape=(40000, 28, 28, 1) aug_y.shape=(40000,)

x = np.concatenate([x_train,aug_x],axis=0)
y = np.concatenate([y_train,aug_y],axis=0)

print(f"{x.shape=} {y.shape=}")
