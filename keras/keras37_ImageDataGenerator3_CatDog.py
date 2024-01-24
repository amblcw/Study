#테스트폴더 쓰진말고 train폴더로
#변환시간도 체크하기

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import time

start_time = time.time()
path = "C:\\_data\\KAGGLE\\cat-and-dog-classification-harper2022\\"
train_path = path+"train\\"
test_path = path+"test\\"

BATCH_SIZE = int(1000)

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
    target_size=(130,130),
    batch_size=BATCH_SIZE,        #batch_size 너무 크게주면 에러나옴
    class_mode='binary',
    shuffle=False
)



# x = np.array(np.arange(0,27)).reshape(3,3,3,1)
# y = np.array(np.arange(0,27)).reshape(3,3,3,1)

# xy = np.vstack([x,y])
# print(x,y,sep='\n')
# print(x.shape,y.shape)
# print(xy,xy.shape) # (6, 3, 3, 1)

import pickle
import os.path

x=[]
y=[]
failed_i = []

data_file_path = path+"\\data.p"
if os.path.isfile(data_file_path):  #파일이 존재하는 여부
    with open(path+"\\data.p",'rb') as file:
        x = pickle.load(file)
        y = pickle.load(file)
        
else:
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

    end_time = time.time()
    print(f"time: {end_time-start_time:.4f}sec")  # time: 284.4365sec


    with open(data_file_path,'wb') as file:
        pickle.dump(x,file)
        pickle.dump(y,file)


    
end_time = time.time()
print(f"time: {end_time-start_time:.4f}sec")  # time: 284.4365sec
print(f"{x.shape=}\n{y.shape=}")    # x.shape=(1000, 200, 200, 3) y.shape=(1000,)


r = int(np.random.uniform(1,1000))
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=r, stratify=y)

# model
model = Sequential()
model.add(Conv2D(32,(2,2),padding='valid',strides=2,input_shape=x_train.shape[1:]))
model.add(MaxPooling2D())
model.add(Conv2D(32,(2,2),padding='valid',strides=2))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Conv2D(32,(2,2),padding='valid',activation='relu'))
model.add(Conv2D(32,(2,2),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Conv2D(64,(2,2),padding='same',activation='relu'))
model.add(Conv2D(64,(2,2),padding='same',activation='relu'))
model.add(BatchNormalization())
# model.add(MaxPooling2D())
# model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# compile & fit
s_time = time.time()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_loss',mode='auto',patience=50,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=1024,batch_size=32,validation_data=(x_test,y_test),verbose=2,callbacks=[es])
e_time = time.time()

# evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)

print(f"loading time: {end_time-start_time:.4f}sec")  # time: 284.4365sec
print(f"fitting time: {e_time-s_time:.4f}sec")
print(f"{r=}\nLOSS: {loss[0]:.6f}\nACC:  {loss[1]:.6f}")
model.save(path+f"model_save\\acc_{loss[1]:.6f}.h5")

# loading time: 31.9445sec
# fitting time: 254.0593sec
# LOSS: 0.396671
# ACC:  0.832124
