from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

#data
path = "C:\\_data\\DACON\\diabets\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0).drop(['Insulin','SkinThickness'],axis=1)
submission_csv = pd.read_csv(path+"sample_submission.csv")

print(train_csv.isna().sum(), test_csv.isna().sum()) # 둘다 결측치 존재하지 않음

test = train_csv['BloodPressure']#.where(train_csv['BloodPressure']==0,train_csv['BloodPressure'].mean())
for i in range(test.size):
    if test[i] == 0:
        test[i] = test.mean()
    # print(test[i])
    
train_csv['BloodPressure'] = test



# print(test)
# zero = train_csv[train_csv['Outcome']==0]
# plt.scatter(range(zero['Insulin'].size),zero['Insulin'],color='red')
# plt.scatter(range(zero['SkinThickness'].size),zero['SkinThickness'],color='blue')
# plt.show()

x = train_csv.drop(['Outcome','Insulin','SkinThickness'],axis=1)
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,shuffle=False)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
# scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model_list = [LinearSVC(), 
              Perceptron(), 
              LogisticRegression(), 
              KNeighborsClassifier(), 
              DecisionTreeClassifier(), 
              RandomForestClassifier(),
              ]
model_names = ['LinearSVC','Perceptron','LogisticRegression','KNeighborsClassifier','DecisionTreeClassifier','RandomForestClassifier']
acc_list = []

for model in model_list:
    #compile & fit
    model.fit(x_train,y_train)

    #evaluate & predict
    acc = round(model.score(x_test,y_test),4)
    # y_predict = model.predict(x_test)
    # acc = accuracy_score(y_test,y_predict)
    acc_list.append(acc)
    
#결과값 출력
print("ACC list: ", acc_list)
print("Best ML: ",model_names[acc_list.index(max(acc_list))])

# ACC list:  [0.93, 0.5, 0.93, 0.93, 0.9, 0.9]
# Best ML:  LinearSVC