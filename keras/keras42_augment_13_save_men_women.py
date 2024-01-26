#https://www.kaggle.com/playlist/men-women-classification
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import time

start_time = time.time()
path = "C:\\_data\\KAGGLE\\men_women\\data"
train_path = path

BATCH_SIZE = int(1000)
IMAGE_SIZE = int(130)

train_data_gen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    # fill_mode='nearest'
)

xy_train_data = train_data_gen.flow_from_directory(
    train_path,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,        #batch_size 너무 크게주면 에러나옴
    class_mode='binary',
    shuffle=False
)

x, y = xy_train_data.next()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=333,stratify=y)


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
    shuffle=False
)

aug_x, aug_y = aug_data.next()

print(f"{aug_x.shape=} {aug_y.shape=}")

aug_x = np.concatenate([x_train,aug_x])
aug_y = np.concatenate([y_train,aug_y])

print(f"{aug_x.shape=}\n{aug_y.shape=}\n{x_test.shape=}\n{y_test.shape=}")    # x.shape=(1000, 200, 200, 3) y.shape=(1000,)

save_path = path+f"data_{IMAGE_SIZE}px"
np.save(save_path+"_aug_x.npy",arr=aug_x)
np.save(save_path+"_aug_y.npy",arr=aug_y)
np.save(save_path+"_test_x.npy",arr=x_test)
np.save(save_path+"_test_y.npy",arr=y_test)

