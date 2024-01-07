#https://dacon.io/competitions/open/235576/overview/description
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import time
import pandas as pd
import math
import datetime


import warnings
warnings.filterwarnings('ignore')

#data
path = "C:\\_data\\DACON\\ddarung\\"
train_csv = pd.read_csv(path+"train.csv",index_col=['id'])  #경로를 적을때 \n같은 경우를 방지하기 위해 \\ 나 /,//도 가능
test_csv = pd.read_csv(path+"test.csv",index_col=0)         #index_col= 을 해줘야 id를 인덱스로 사용, 0 또는 헤더명
submission_csv = pd.read_csv(path+"submission.csv")

# print(f"{train_csv.shape=}\n{test_csv.shape=}\n{submission_csv.shape=}")
# train_csv.shape=(1459, 10)
# test_csv.shape=(715, 9)
# submission_csv.shape=(715, 2)


#### 결측치 처리 1. 제거 #### 나중에 드랍 말고 다른 것도 적용해보기
# print(train_csv.isnull().sum()) 아래와 같음
# print(train_csv.isna().sum())
# print(test_csv.isna().sum())


# train_csv = train_csv.dropna()

nan_index = np.where(np.isnan(train_csv))

row, colum = nan_index

for i in range(len(row)):
    # print(f"({row[i]},{colum[i]})",train_csv.iloc[row[i],colum[i]])
    pre = train_csv.iloc[row[i]-1,colum[i]]
    next = train_csv.iloc[row[i]+1,colum[i]]
    train_csv.iloc[row[i],colum[i]] = (pre + next)/2
    # print(f"({row[i]},{colum[i]})",train_csv.iloc[row[i],colum[i]])

nan_index2 = np.where(np.isnan(test_csv))

row2, colum2 = nan_index2

for i in range(len(row2)):
    # print(f"({row[i]},{colum[i]})",train_csv.iloc[row[i],colum[i]])
    pre = test_csv.iloc[row2[i]-1,colum2[i]]
    next = test_csv.iloc[row2[i]+1,colum2[i]]
    test_csv.iloc[row2[i],colum2[i]] = (pre + next)/2

# print(train_csv.isna().sum(), test_csv.isna().sum())


train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())
# train_csv = train_csv.fillna(0)
# test_csv = test_csv.fillna(0)



# print(train_csv.isna().sum(), test_csv.isna().sum())

x = train_csv.drop(['count'],axis=1) #count 를 드랍, axis=0은 행, axis=1은 열
y = train_csv['count']

loss_list = []
r2_list = []
submit_list = []

r2=0
# for i in range(10):
# while r2<0.69:
r = int(np.random.uniform(1,1000))
r = 285
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=r)
print(f"{x_train.shape=} , {y_train.shape=}")   #x_train.shape=(1167, 9) , y_train.shape=(1167,)

# model
model = Sequential()
# model.add(Dense(1,input_dim=9))
model.add(Dense(16,input_dim=9))
# model.add(Dense(32))
# model.add(Dense(48))
# model.add(Dense(64))
# model.add(Dense(96))
# model.add(Dense(128))
# model.add(Dense(198))
# model.add(Dense(256))
# model.add(Dense(198))
# model.add(Dense(128))
# model.add(Dense(96))
# model.add(Dense(64))
# model.add(Dense(48))
# model.add(Dense(32))
# model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#compile & fit
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1024,batch_size=4,verbose=2)

#evaluate & predict
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
y_submit = model.predict(test_csv)

print(f"{r=}\nRMSE: {math.sqrt(loss)}\n{r2=}")
time.sleep(1.5)


# print(y_submit)
# print(y_submit.shape)   #(751, 1)
loss_list.append(loss)
r2_list.append(r2)
submit_list.append(y_submit)

print(loss_list,r2_list,sep='\n')
print(f"LOSS mean: {np.mean(loss_list)}\nR2 mean: {np.mean(r2_list)}")

#### csv 만들기 ####

submission_csv['count'] = submit_list[r2_list.index(max(r2_list))]
# print(submission_csv)

dt = datetime.datetime.now()
# print(dt.day,dt.hour,dt.minute)
submission_csv.to_csv(path+f"submission_{dt.day}day{dt.hour}-{dt.minute}.csv",index=False)

# r=97
# RMSE: 45.58303131803133
# r2=0.6803156365354164

# r=376
# RMSE: 46.712875884893876
# r2=0.6760600687694152

#8421 레이어일때
# [2151.736328125, 2114.722412109375, 2152.9267578125, 2164.705322265625, 2365.150634765625, 2164.657470703125, 2196.457275390625, 2100.888427734375, 2279.8896484375, 2114.105224609375]
# [0.6689420354910287, 0.6746368705477295, 0.6687588645720783, 0.6669466906482122, 0.6361068944783301, 0.666954004604711, 0.6620614439048904, 0.6767652830767787, 0.6492248399099415, 0.6747318015761978]
# LOSS mean: 2180.5239501953124
# R2 mean: 2180.5239501953124

#48이 피크일 때
# [2382.630615234375, 2104.373779296875, 2079.283935546875, 2086.9423828125, 2303.2373046875, 2092.30078125, 2333.1494140625, 2103.80126953125, 2081.216796875, 2255.657958984375]
# [0.6334175370709949, 0.6762290538295346, 0.6800892792484833, 0.6789109792515216, 0.6456326496207925, 0.6780865517217547, 0.6410305392694677, 0.6763171664691928, 0.6797918753875232, 0.6529530734817269]
# LOSS mean: 2182.259423828125
# R2 mean: 0.6642458705350992

#128 피크일 때 
# [2342.38818359375, 2191.28466796875, 2100.66943359375, 2118.937255859375, 2247.3359375, 2320.377685546875, 2264.223876953125, 2301.606689453125, 2162.61376953125, 2266.310791015625]
# [0.6396090119417261, 0.6628572487896764, 0.676798953798605, 0.6739883625936846, 0.6542334667758893, 0.6429955482396517, 0.6516351328402468, 0.6458835482395409, 0.6672685045971449, 0.6513140217236046]
# LOSS mean: 2231.5748291015625
# R2 mean: 0.656658379953977

#8421레이어 1024회 훈련
# [2067.8271484375, 2094.36083984375, 2052.67333984375, 2117.114013671875, 2057.5361328125, 2074.112060546875, 2077.8671875, 2074.443603515625, 2066.2978515625, 2074.63525390625]
# [0.6818519664927092, 0.6777695788563591, 0.6841834822383354, 0.6742688809328556, 0.6834353214622559, 0.6808850086508238, 0.6803072467981431, 0.680834011104618, 0.6820872999065146, 0.6808045040107468] 
# LOSS mean: 2075.6867431640626
# R2 mean: 0.6806427300453362

#8421레이어 2048회 훈련
# [2174.57763671875, 2067.84814453125, 2111.7958984375, 2044.9608154296875, 2075.06640625, 2110.235107421875, 2073.840087890625, 2088.180419921875, 2168.029052734375, 2075.403564453125]
# [0.6654277835203164, 0.681848736524346, 0.6750870976179174, 0.6853701257182265, 0.680738197932573, 0.6753272908454534, 0.6809268610012049, 0.6787204867731416, 0.6664353027665137, 0.6806863051852762]  
# LOSS mean: 2098.9937133789062
# R2 mean: 0.6770568187884969

#88421레이어 512회 훈련, batch_size=4
# RMSE: 45.34807601262566
# r2=0.6836027245280307
# [2056.447998046875]
# [0.6836027245280307]
# LOSS mean: 2056.447998046875

#동일 조건, Nan 처리를 drop으로 했을 때
# [3606.00732421875]
# [0.5180706974347669]
# LOSS mean: 3606.00732421875
# R2 mean: 0.5180706974347669

# train 비중을 0.7로 낮췄을 때
# RMSE: 53.0158437403386
# r2=0.5774311018227056
# [2810.6796875]
# [0.5774311018227056]
# LOSS mean: 2810.6796875
# R2 mean: 0.5774311018227056

#fillna(0)
# RMSE: 45.51148657378981
# r2=0.6813183622943629
# [2071.29541015625]
# [0.6813183622943629]
# LOSS mean: 2071.29541015625
# R2 mean: 0.6813183622943629

#fillna(0) 2회차
# RMSE: 49.15514491367485
# r2=0.6282482922552581
# [2416.228271484375]
# [0.6282482922552581]
# LOSS mean: 2416.228271484375
# R2 mean: 0.6282482922552581

#훈련 횟수 128로 줄여보기
# RMSE: 70.34018552164012
# r2=0.2387593048768234
# [4947.74169921875]
# [0.2387593048768234]
# LOSS mean: 4947.74169921875
# R2 mean: 0.2387593048768234

# Traceback (most recent call last):
#   File "<frozen runpy>", line 198, in _run_module_as_main
#   File "<frozen runpy>", line 88, in _run_code    
#   File "c:\Users\wlily\.vscode\extensions\ms-python.python-2023.22.1\pythonFiles\lib\python\debugpy\adapter/../..\debugpy\launcher\__main__.py", line 
# 91, in <module>
#     main()
#   File "c:\Users\wlily\.vscode\extensions\ms-python.python-2023.22.1\pythonFiles\lib\python\debugpy\adapter/../..\debugpy\launcher\__main__.py", line 
# 47, in main
#     launcher.connect(host, port)
#   File "c:\Users\wlily\.vscode\extensions\ms-python.python-2023.22.1\pythonFiles\lib\python\debugpy\adapter/../..\debugpy\launcher/../..\debugpy\launcher\__init__.py", line 27, in connect
#     sock.connect((host, port))
# OSError: [WinError 10049] 요청한 주소는 해당 컨텍 
# 스트에서 유효하지 않습니다

#86421레이어
# RMSE: 45.35227510745051
# r2=0.683544161425371
# [2056.828857421875]
# [0.683544161425371]
# LOSS mean: 2056.828857421875
# R2 mean: 0.683544161425371

#위와 동일, 배치사이즈 32로
# [2098.32568359375, 2125.633544921875, 2091.154541015625, 2084.49072265625, 2099.593994140625, 2080.016357421875, 2112.98291015625, 2204.400634765625, 2064.084228515625, 2096.63232421875]
# [0.677159593163063, 0.6729581135554364, 0.678262938757972, 0.6792881791896368, 0.6769644741418188, 0.6799766012497923, 0.6749045161725038, 0.6608393051206192, 0.6824278428978767, 0.6774201465101761]
# LOSS mean: 2105.731494140625
# R2 mean: 0.6760201710758895

#어떻게 해도 처음이 제일 좋으니 처음 조건 그대로 다시 다만 R2 값은 오히려 낮았기에 R2값으로 판단 불가, 전부 올려서 테스트해보기
# RMSE: 54.526629460406774
# r2=0.6203538588450952
# [2973.1533203125]
# [0.6203538588450952]

# RMSE: 54.542009593953125
# r2=0.620139696021671
# [2974.830810546875]
# [0.620139696021671]

# RMSE: 54.02795364800174
# r2=0.6272662178842985
# [2919.019775390625]
# [0.6272662178842985]

# RMSE: 53.92531592179711
# r2=0.6286810640834528
# [2907.939697265625]
# [0.6286810640834528]

#batch_size를 오히려 16으로 늘렸을 때
# RMSE: 55.36393045044648
# r2=0.6086047510512438
# [3065.164794921875]
# [0.6086047510512438]

# 8421레이어
#RMSE: 54.061398110158045
# r2=0.6268046350786065
# [2922.634765625]
# [0.6268046350786065]

#다시 초기 레이어
# RMSE: 55.55196662669783
# r2=0.6059415976425613
# [3086.02099609375]
# [0.6059415976425613]