#테스트폴더 쓰진말고 train폴더로
#변환시간도 체크하기

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping
import numpy as np
import time

start_time = time.time()
path = "C:\\_data\\KAGGLE\\cat-and-dog-classification-harper2022\\"
train_path = path+"train\\"
test_path = path+"test\\"

BATCH_SIZE = int(100)

train_data_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

xy_train_data = train_data_gen.flow_from_directory(
    train_path,
    target_size=(200,200),
    batch_size=BATCH_SIZE,        #batch_size 너무 크게주면 에러나옴
    class_mode='binary',
    shuffle=False
)

end_time = time.time()
print(f"time: {end_time-start_time:.4f}sec")

# x = np.array(np.arange(0,27)).reshape(3,3,3,1)
# y = np.array(np.arange(0,27)).reshape(3,3,3,1)

# xy = np.vstack([x,y])
# print(x,y,sep='\n')
# print(x.shape,y.shape)
# print(xy,xy.shape) # (6, 3, 3, 1)

x=[]
y=[]
failed_i = []

for i in range(int(20000 / BATCH_SIZE)):
    try: 
        xy_data = xy_train_data.next()
        new_x = xy_data[0]
        new_y = xy_data[1]
        if i==0:
            x = np.array(new_x)
            y = np.array(new_y)
            continue
        
        x = np.vstack([x,new_x])
        y = np.hstack([y,new_y])
        print("i: ",i)
        print(f"{x.shape=}\n{y.shape=}")
    except:
        print("failed i: ",i)
        failed_i.append(i)
        
print(failed_i) # [70, 115]

print(f"{x.shape=}\n{y.shape=}")    # x.shape=(1000, 200, 200, 3) y.shape=(1000,)

# 파일탐색기 - 사진크기순 분류 - 맨 밑으로 내리면 지정되지 않음 존재
# 불량파일: cat/666.jpg cat/10404.jpg dog/11702.jpg