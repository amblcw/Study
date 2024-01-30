import numpy as np
from function_package import split_x as split2
from function_package import split_xy
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

a = np.array(range(1,101))
x_predict = np.array(range(96,106))
size = 4                                # x데이터는 4개, y데이터는 1개

def split_x(dataset, size):             # dataset 과 size를 인자로 받는다, dataset은 리스트이며 size는 정수여야한다
    aaa = []                            # 반환하기 위한 임시 리스트 
    for i in range(len(dataset)-size+1):# dataset을 size만큼 자르면 len(dataset) - size + 1 만큼 자를 수있다
        subset = dataset[i: (i+size)]   # dataset을 i번 부터 size만큼 자른다, i 는 0부터 len(dataset)-size 까지의 정수이다
        aaa.append(subset)              # 잘라낸 데이터를 aaa에 붙인다
        
    return np.array(aaa)                # 완성된 리스트를 반환한다


x, y = split_xy(a,size)
x_predict = split2(x_predict,size)

x = x.reshape(-1,4)
x_predict = x_predict.reshape(-1,4)

print(x.shape,x_predict.shape,y.shape)  # (96, 4) (7, 4) (96,)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=333)

model = Sequential()
# model.add(LSTM(512,input_shape=(4,1),return_sequences=False,activation='relu'))
model.add(Dense(512,input_dim=4,activation='relu'))
model.add(Dense(16))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
es = EarlyStopping(monitor='val_loss',mode='auto',patience=100,restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=8192, batch_size=8, validation_data=(x_test,y_test),verbose=2,callbacks=[es])

loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_predict)

for data in x_predict:
    data = data.reshape(-1,)
    print(data)
print(f"LOSS: {loss}")
print(f"y_predict= \n{y_predict}")

# [96 97 98 99]
# [ 97  98  99 100]
# [ 98  99 100 101]
# [ 99 100 101 102]
# [100 101 102 103]
# [101 102 103 104]
# [102 103 104 105]
# LOSS: 1.4165231732476968e-05
# y_predict=
# [[100.0109  ]
#  [101.01681 ]
#  [102.015175]
#  [103.02071 ]
#  [104.0224  ]
#  [105.02577 ]
#  [106.02608 ]]

# DNN
# [96 97 98 99]
# [ 97  98  99 100]
# [ 98  99 100 101]
# [ 99 100 101 102]
# [100 101 102 103]
# [101 102 103 104]
# [102 103 104 105]
# LOSS: 2.2188650916632469e-07
# y_predict=
# [[100.00064 ]
#  [101.000656]
#  [102.00069 ]
#  [103.00072 ]
#  [104.00072 ]
#  [105.00069 ]
#  [106.000656]]