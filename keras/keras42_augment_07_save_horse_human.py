from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from function_package import merge_image

BATCH_SIZE = int(500)
IMAGE_SIZE = int(300)   #변경금지 시험 요구사항

path = "C:\\_data\\image\\horse-or-human\\"

xy_data_gen = ImageDataGenerator(
    rescale=1./255,
)

xy_data_b = xy_data_gen.flow_from_directory(
    path,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

x_b, y_b = merge_image(xy_data_b)

# randidx = np.random.randint(x_b.shape[0],size=20000)

# splited_x = x_b[randidx].copy()
# splited_y = y_b[randidx].copy()

splited_x = np.concatenate([x_b,x_b,x_b],axis=0)
splited_y = np.concatenate([y_b,y_b,y_b],axis=0)

xy_augment_data_gen = ImageDataGenerator(
    vertical_flip=True,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=20,
    shear_range=20,
    fill_mode='nearest'
)

xy_augment_data = xy_augment_data_gen.flow(
    splited_x,
    splited_y,
    batch_size=BATCH_SIZE,
    shuffle=False
)

aug_x, aug_y = merge_image(xy_augment_data)

extend_x = np.concatenate([x_b,aug_x],axis=0)
extend_y = np.concatenate([y_b,aug_y],axis=0)

print(f"{extend_x.shape=} {extend_y.shape=}")

save_path = path+f"data_{IMAGE_SIZE}px"
np.save(save_path+"_aug_x.npy",arr=extend_x)
np.save(save_path+"_aug_y.npy",arr=extend_y)
print(f"data_{IMAGE_SIZE}px_b file created")