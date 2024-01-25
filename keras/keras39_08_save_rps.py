from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from function_package import merge_image    #같은 keras폴더 내의 function_package 참조

'''
image_size는 꼭 150,150,3으로 즉 컬러로 해야한다
'''

BATCH_SIZE = int(500)
IMAGE_SIZE = int(150)

path = "C:\\_data\\image\\rps\\"



xy_data_gen = ImageDataGenerator(
    rescale=1./255,
)
xy_data = xy_data_gen.flow_from_directory(
    path,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
)

x, y = merge_image(xy_data)

save_path = path+f"data_{IMAGE_SIZE}px"
np.save(save_path+"_x.npy",arr=x)
np.save(save_path+"_y.npy",arr=y)

