#테스트폴더 쓰진말고 train폴더로
#변환시간도 체크하기

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import time
from function_package import merge_image

start_time = time.time()
path = "C:\\_data\\etc\\brain\\"
train_path = path+"train\\"
test_path = path+"test\\"

BATCH_SIZE = int(200)
IMAGE_SIZE = int(200)

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

xy_train = train_data_gen.flow_from_directory(
    train_path,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,        #batch_size 너무 크게주면 에러나옴
    class_mode='binary',
    shuffle=False
    
)


test_data_gen = ImageDataGenerator( #테스트는 실제 데이터여야 의미가 있기에 변형을 하지 않는다
    rescale=1./255,
)

xy_test = test_data_gen.flow_from_directory(
    test_path,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False,
)

x_train, y_train = (xy_train[0][0], xy_train[0][1])
x_test, y_test = (xy_test[0][0], xy_test[0][1])

amplify_num = 20

amplified_xy_data_gen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    fill_mode='nearest'
)

amplified_xy_data = amplified_xy_data_gen.flow(
    x_train,
    y_train,
    batch_size=x_train.shape[0],
    shuffle=False
)

amplified_x = []
amplified_y = []
for i in range(amplify_num):
    amp_x, amp_y = amplified_xy_data.next()
    amplified_x.append(amp_x)
    amplified_y.append(amp_y)

amplified_x = np.concatenate(amplified_x)
amplified_y = np.concatenate(amplified_y)

x_train = np.concatenate([x_train,amplified_x])
y_train = np.concatenate([y_train,amplified_y])

print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")
# x_train.shape=(160, 200, 200, 3)
# x_test.shape=(120, 200, 200, 3)
# y_train.shape=(160,)
# y_test.shape=(120,)

np_path = "C:\\_data\\_save_npy\\"
np.save(np_path+f"keras39_1_x_train.npy",arr=x_train)
np.save(np_path+f"keras39_1_y_train.npy",arr=y_train)
np.save(np_path+f"keras39_1_x_test.npy",arr=x_test)
np.save(np_path+f"keras39_1_y_test.npy",arr=y_test)

print("end")