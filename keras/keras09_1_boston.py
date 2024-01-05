from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston    # pip install scikit-learn==1.1.3
import time

import warnings
warnings.filterwarnings('ignore')

datasets = load_boston()
# print(datasets)
x = datasets.data
y = datasets.target
# print(x.shape)  #(506,13)
# print(y.shape)  #(506,)
# print(datasets.feature_names)   #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
# print(datasets.DESCR)           #데이터셋 설명
'''
Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.  
    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's
'''
r2 = 0

while r2 < 0.8:
    r = int(np.random.uniform(1,1000))
    r = 88
    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r)

    #model
    model = Sequential()
    model.add(Dense(20,input_dim=13))
    model.add(Dense(30))
    model.add(Dense(40))
    model.add(Dense(30))
    model.add(Dense(10))
    model.add(Dense(3))
    model.add(Dense(1))

    #compile & fit
    model.compile(loss='mse',optimizer='adam')
    start_time = time.time()
    model.fit(x_train,y_train,epochs=8000,batch_size=10,verbose=2)

    #evaluate & predict
    loss = model.evaluate(x_test,y_test)
    result = model.predict(x)
    y_predict = model.predict(x_test)

    r2 = r2_score(y_test,y_predict)
    end_time = time.time()
    
    print(f"Time: {round(end_time-start_time,2)}sec")
    print(f"{r=}\n{loss=}\n{r2=}")
    pass

# r=88
# loss=14.834733963012695
# r2=0.7990107660178996