from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from function_package import merge_image

BATCH_SIZE = int(500)
IMAGE_SIZE = int(300)   #변경금지 시험 요구사항

path = "C:\\_data\\image\\horse-or-human\\"

xy_data_gen = ImageDataGenerator(
    rescale=1./255,
)

# xy_data = xy_data_gen.flow_from_directory(
#     path,
#     target_size=(IMAGE_SIZE,IMAGE_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     shuffle=False
# )


# x = []
# y = []
# failed_i = []

# for i in range(len(xy_data)):
#     try:
#         xy = xy_data.next()
#         new_x = xy[0]
#         new_y = xy[1]
#         if i == 0:
#             x = np.array(new_x)
#             y = np.array(new_y)
#             continue
        
#         x = np.vstack([x,new_x])
#         y = np.vstack([y,new_y])
#         print("i: ",i)
#         print(f"{x.shape=}\n{y.shape=}")
#     except:
#         print("failed i: ",i)
#         failed_i.append(i)
        
# print(failed_i)
# print(f"{x.shape=}\n{y.shape=}")

# save_path = path+f"data_{IMAGE_SIZE}px"
# np.save(save_path+"_x.npy",arr=x)
# np.save(save_path+"_y.npy",arr=y)


xy_data_b = xy_data_gen.flow_from_directory(
    path,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

x_b, y_b = merge_image(xy_data_b)

save_path = path+f"data_{IMAGE_SIZE}px"
np.save(save_path+"_x_b.npy",arr=x_b)
np.save(save_path+"_y_b.npy",arr=y_b)
print(f"data_{IMAGE_SIZE}px_b file created")