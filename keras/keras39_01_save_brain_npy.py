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

data = np.arange([[xy_train[0][0],xy_train[0][1]],[xy_test[0][0],xy_test[0][1]]])

print(data.shape)

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
# import pickle
# import os.path

# x=[]
# y=[]
# failed_i = []
# data_file_path = path+f"\\data_image{IMAGE_SIZE}px.p"

# if os.path.isfile(data_file_path):  #파일이 존재하는 여부
#     with open(data_file_path,'rb') as file:
#         x = pickle.load(file)
#         y = pickle.load(file)
        
# else:
#     for i in range(int(20000 / BATCH_SIZE)):
#         try: 
#             xy_data = xy_train.next()
#             new_x = xy_data[0]
#             new_y = xy_data[1]
#             if i==0:
#                 x = np.array(new_x)
#                 y = np.array(new_y)
#                 continue
            
#             x = np.vstack([x,new_x])
#             y = np.hstack([y,new_y])
#             print("i: ",i)
#             print(f"{x.shape=}\n{y.shape=}")
#         except:
#             print("failed i: ",i)
#             failed_i.append(i)
            
#     print(failed_i) # [70, 115]

#     print(f"{x.shape=}\n{y.shape=}")    # x.shape=(1000, 200, 200, 3) y.shape=(1000,)
#     # 파일탐색기 - 사진크기순 분류 - 맨 밑으로 내리면 지정되지 않음 존재
#     # 불량파일: cat/666.jpg cat/10404.jpg dog/11702.jpg

#     end_time = time.time()
#     print(f"time: {end_time-start_time:.4f}sec")  # time: 284.4365sec


#     with open(data_file_path,'wb') as file:
#         pickle.dump(x,file)
#         pickle.dump(y,file)


    
# end_time = time.time()
# print(f"time: {end_time-start_time:.4f}sec")  # time: 284.4365sec
# print(f"{x.shape=}\n{y.shape=}")    # x.shape=(1000, 200, 200, 3) y.shape=(1000,)


# r = int(np.random.uniform(1,1000))
# r = 965
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=r, stratify=y)

# .836755 모델
# model = Sequential()
# model.add(Conv2D(32,(3,3),padding='valid',strides=2,input_shape=x_train.shape[1:]))
# model.add(MaxPooling2D())
# model.add(Conv2D(32,(3,3),padding='valid',strides=2))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
# model.add(Dropout(0.15))
# model.add(Conv2D(32,(2,2),padding='valid',activation='relu'))
# model.add(Conv2D(32,(2,2),padding='valid',activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
# model.add(Dropout(0.15))
# model.add(Conv2D(64,(2,2),padding='same',activation='relu'))
# model.add(Conv2D(64,(2,2),padding='same',activation='relu'))
# model.add(BatchNormalization())
# # model.add(MaxPooling2D())
# # model.add(Dropout(0.15))
# model.add(Flatten())
# model.add(Dense(1024,activation='relu'))
# model.add(Dense(128,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))
'''
model = Sequential()
model.add(Conv2D(32,(3,3),padding='valid',strides=2,input_shape=x_train.shape[1:]))
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3),padding='valid',strides=2))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Conv2D(32,(3,3),padding='valid',activation='relu'))
model.add(Conv2D(32,(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Conv2D(64,(2,2),padding='valid',activation='relu'))
model.add(Conv2D(64,(2,2),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# compile & fit
s_time = time.time()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_loss',mode='auto',patience=100,restore_best_weights=True)
hist = model.fit(
    xy_train,               #x_train, y_train 대신 이렇게 iterator로 해줄수있다, 이 경우 batch_size를 IDG에서 결정한다
    # batch_size=32         #이건 무시된다, fit_generator에선 에러 나옴
    steps_per_epoch=16,     #1epo 마다 실행되는 횟수, data / batch_size = step_per_epochs 더 크면 에러, 작으면 데이터 손실난다
    epochs=1024,
    validation_data=xy_test,#split먹히지 않고 이걸 써야한다
    # validation_split=0.2  #error, split은 오직 np.array와 Tenser에서만 쓸 수 있다 iterator에선 불가능
    verbose=2,
    callbacks=[es])
e_time = time.time()

# evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)

print(f"loading time: {end_time-start_time:.4f}sec")  # time: 284.4365sec
print(f"fitting time: {e_time-s_time:.4f}sec")
print(f"LOSS: {loss[0]:.6f}\nACC:  {loss[1]:.6f}")
model.save(path+f"model_save\\acc_{loss[1]:.6f}.h5")

import matplotlib.pyplot as plt

if hist != []:
    plt.title("Cat&Dog CNN")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.plot(hist.history['val_acc'],label='val_acc',color='red')
    plt.plot(hist.history['acc'],label='acc',color='blue')
    # plt.plot(hist.history['val_loss'],label='val_loss',color='red')
    # plt.plot(hist.history['loss'],label='loss',color='blue')
    plt.legend()
    plt.show()

# loading time: 0.3882sec
# fitting time: 34.4479sec
# LOSS: 0.009368
# ACC:  1.000000
'''