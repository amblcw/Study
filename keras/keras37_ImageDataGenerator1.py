from keras.models import Sequential
from keras.layers import Conv2D, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np



train_data_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,   # 수평으로 뒤집는다.
    vertical_flip=True,     # 수직으로 뒤집는다
    width_shift_range=0.1,  # 0.1(10%)만큼 수평이동, 이동한 공백은 0으로 채움
    height_shift_range=0.1, # 0.1(10%)만큼 수직이동
    rotation_range=5,       # 이 각도만큼 이미지를 회전
    zoom_range=1.2,         # 1.2배 확대
    shear_range=0.7,        
    
    fill_mode='nearest'     # 빈공간을 무엇으로 채울지 nearest는 맨 끝값으로 채움 ex)'nearest': aaaaaaaa|abcd|dddddddd
)

test_data_gen = ImageDataGenerator( #테스트는 실제 데이터여야 의미가 있기에 변형을 하지 않는다
    rescale=1./255,
)

path_train = "C:\\_data\\etc\\brain\\train\\"
path_test = "C:\\_data\\etc\\brain\\test\\"

xy_train = train_data_gen.flow_from_directory(
    path_train,
    target_size=(200,200),
    batch_size=10,          #batch_size만큼 잘라서 인덱스에 담아주지만 통짜로 잘라서 하고싶으면 160(모든데이터 개수)을 입력하면된다
    # batch_size=160,         
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)
# print(xy_train)
# Found 160 images belonging to 2 classes.
# <keras.preprocessing.image.DirectoryIterator object at 0x000002100F0D0670>
# print(xy_train.next(),xy_train[0])  
# iterator.next()는 다음 하나의 데이터를 보여줌
# xy_train[0]에는 batch_size만큼 이미지가 담겨있으며, y값도 같이 담겨있음
# print(xy_train[16])#<- error, 총 데이터 160개에서 batch_size만큼인 10개씩 담겼으므로 마지막 인덱스는 15
# print(xy_train[0][1])     #xy_train[0][0]은 첫번째 배치의 x값, xy_train[0][1]은 첫번째 배치의 y값
print(xy_train[0][0].shape) #(10, 200, 200, 1)
print(type(xy_train))       #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    #<class 'tuple'>
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
print(type(xy_train[0][1])) #<class 'numpy.ndarray'>

xy_test = test_data_gen.flow_from_directory(
    path_test,
    target_size=(200,200),
    batch_size=10,
    class_mode='binary',
    shuffle=False,
)
# print(xy_test)
# Found 120 images belonging to 2 classes.
# <keras.preprocessing.image.DirectoryIterator object at 0x000001D0C0542E50>
print()