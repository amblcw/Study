from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from function_package import merge_image    #같은 keras폴더 내의 function_package 참조
from sklearn.model_selection import train_test_split

'''
image_size는 꼭 150,150,3으로 즉 컬러로 해야한다
'''

BATCH_SIZE = int(500)
IMAGE_SIZE = int(150)

path = "C:\\_data\\image\\rps\\"



xy_data_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest'
)
xy_data = xy_data_gen.flow_from_directory(
    path,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
)

x, y = merge_image(xy_data)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=333, stratify=y)

amplified_x, amplified_y = np.concatenate([x_train,x_train,x_train]), np.concatenate([y_train,y_train,y_train])

amplified_xy_data_gen = ImageDataGenerator(
    # horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=20,
    shear_range=20,
    fill_mode='nearest'
)

amplified_xy_data = amplified_xy_data_gen.flow(
    amplified_x,
    amplified_y,
    batch_size=BATCH_SIZE,
    shuffle=False
)

amplified_x, amplified_y = merge_image(amplified_xy_data)

x, y = np.concatenate([x_train,amplified_x]), np.concatenate([y_train,amplified_y])

print(f"{x.shape=},{y.shape=}")

save_path = path+f"data_{IMAGE_SIZE}px"
np.save(save_path+"_aug_x.npy",arr=x)
np.save(save_path+"_aug_y.npy",arr=y)
np.save(save_path+"_test_x.npy",arr=x_test)
np.save(save_path+"_test_y.npy",arr=y_test)

