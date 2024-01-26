from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img     #이미지 불러오기
from keras.utils import img_to_array #이미지를 수치화
import numpy as np
from function_package import merge_image    #같은 keras폴더 내의 function_package 참조
import matplotlib.pyplot as plt

'''
image_size는 꼭 150,150,3으로 즉 컬러로 해야한다
'''
BATCH_SIZE = int(500)
IMAGE_SIZE = int(150)

path = "C:\\_data\\KAGGLE\\cat-and-dog-classification-harper2022\\train\\Cat"

img = load_img(
    path+"\\1.jpg",
    target_size=(IMAGE_SIZE,IMAGE_SIZE)
    )

print(img)      #<PIL.Image.Image image mode=RGB size=150x150 at 0x14C857BD160>
print(type(img))#<class 'PIL.Image.Image'>
# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
# print(arr)
print(f"{arr.shape=},{type(arr)}")  # arr.shape=(281, 300, 3) -> arr.shape=(150, 150, 3),<class 'numpy.ndarray'>

# 차원증가
img = np.expand_dims(arr, axis=0)
print(img.shape)

##################################### 증폭구간 ##################################### 
data_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.3,
    shear_range=20,
    rotation_range=20,
    channel_shift_range=20,
    fill_mode='reflect'
    )

it = data_gen.flow(
    img,
    batch_size=1,
    )

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(16,9))

for row in range(2):
    for col in range(5):
        batch = it.next()
        image=batch[0]
        
        ax[row][col].imshow(image)
        ax[row][col].axis('off')
    
plt.show()
