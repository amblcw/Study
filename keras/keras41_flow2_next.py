from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(f"{x_train.shape=}, {y_train.shape=}, {x_test.shape=}, {y_test.shape=}")
# x_train.shape=(60000, 28, 28), y_train.shape=(60000,), x_test.shape=(10000, 28, 28), y_test.shape=(10000,)

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

AUG_SIZE = 10
augment_size = AUG_SIZE*AUG_SIZE

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1),    # x
    np.zeros(augment_size),                                                 # y
    batch_size=augment_size,
    shuffle=False
)

x, y = x_data.next()
# print(x[0].shape)
# print(np.unique(y,return_counts=True))

plt.figure(figsize=(AUG_SIZE,AUG_SIZE))
for i in range(augment_size):
    plt.subplot(AUG_SIZE,AUG_SIZE,i+1)
    plt.axis('off')
    plt.imshow(x[i],'gray')

# fig , ax = plt.subplots(nrows=AUG_SIZE,ncols=AUG_SIZE, figsize=(16,9))
# for row in range(AUG_SIZE):
#     for col in range(AUG_SIZE):
#         ax[row][col].imshow(x[row*col],'gray')
#         ax[row][col].axis('off')

plt.show()