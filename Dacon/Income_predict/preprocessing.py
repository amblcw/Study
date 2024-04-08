import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import random

# RandomState:  42250
RANDOM_STATE = random.randint(1,10000)
tf.random.set_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

PATH = 'C:/Study/Dacon/Income_predict/'

train_csv = pd.read_csv(PATH+'train.csv', index_col=0)
test_csv = pd.read_csv(PATH+'test.csv',index_col=0)
submit_csv = pd.read_csv(PATH+'sample_submission.csv')

print(train_csv.shape, train_csv.isna().sum())  # (20000, 22) 결측치 존재 안함
print(test_csv.shape, test_csv.isna().sum())    # (10000, 21) Household_Status에 1개 존재

print(test_csv[test_csv['Household_Status'].isna()])    # TEST_2659
test_csv = test_csv.fillna(method='bfill')      # 결측치 제거, 뒤쪽 값으로 채운 이유는 그나마 비슷해서
print(test_csv.shape, test_csv.isna().sum())    # (10000, 21) 결측치 존재 안함

print(train_csv.columns)
# Index(['Age', 'Gender', 'Education_Status', 'Employment_Status',
#        'Working_Week (Yearly)', 'Industry_Status', 'Occupation_Status', 'Race',
#        'Hispanic_Origin', 'Martial_Status', 'Household_Status',
#        'Household_Summary', 'Citizenship', 'Birth_Country',
#        'Birth_Country (Father)', 'Birth_Country (Mother)', 'Tax_Status',
#        'Gains', 'Losses', 'Dividends', 'Income_Status', 'Income'],
#       dtype='object')

        
# 출신국가 미국과 미국 이외로 이원화
train_csv.loc[train_csv['Birth_Country'] != 'US', 'Birth_Country'] = 'not US'
train_csv.loc[train_csv['Birth_Country (Father)'] != 'US', 'Birth_Country (Father)'] = 'not US'
train_csv.loc[train_csv['Birth_Country (Mother)'] != 'US', 'Birth_Country (Mother)'] = 'not US'

test_csv.loc[test_csv['Birth_Country'] != 'US', 'Birth_Country'] = 'not US'
test_csv.loc[test_csv['Birth_Country (Father)'] != 'US', 'Birth_Country (Father)'] = 'not US'
test_csv.loc[test_csv['Birth_Country (Mother)'] != 'US', 'Birth_Country (Mother)'] = 'not US'

# 라벨 범위 동일한지 확인
for label in test_csv:
    train_data = train_csv[label].copy()
    test_data = test_csv[label].copy()
    if test_data.dtypes == 'object':
        train_labels = set(np.unique(train_data))
        test_labels = set(np.unique(test_data))

        diff = train_labels.difference(test_labels)
        if len(diff) != 0:
            print(diff)
            break
else:
    print("train_csv와 test_csv의 모든 columns에서 라벨은 전부 동일합니다")        
# {'Grandchild 18+ spouse of subfamily Responsible Person', 'Other Relative <18 ever married Responsible Person of subfamily', 'Child <18 ever married Responsible Person of subfamily', 'Grandchild 18+ ever married Responsible Person of subfamily'}
# 라벨이 서로 같지 않음

   
# 라벨 인코딩
from sklearn.preprocessing import LabelEncoder
label_encoder_dict = {}
for label in train_csv:
    data = train_csv[label].copy()
    if data.dtypes == 'object':
        label_encoder = LabelEncoder()
        train_csv[label] = label_encoder.fit_transform(data)
        label_encoder_dict[label] = label_encoder
  
""" inverse_transform 작동 확인 완료
for label, label_encoder in label_encoder_dict.items():
    data = train_csv[label].copy()
    train_csv[label] = label_encoder.inverse_transform(data)
print(train_csv.head(10))
"""

for label, encoder in label_encoder_dict.items():
    data = test_csv[label]
    test_csv[label] = encoder.transform(data)
# print(test_csv.head(10))
# print(test_csv.isna().sum())


x = train_csv.drop(['Income'],axis=1)
y = train_csv['Income']


# 이상치 확인
target_label = ['Gains','Losses','Dividends']
for label in x:
    data = x[label]
    # data = data[data != 0]
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = (q3-q1) * 3.0
    upper = q3+iqr
    under = q1-iqr
    over_outlier = data[data > upper]
    below_outlier = data[data < under]
    print(label, ": ",over_outlier,below_outlier)
    plt.boxplot(data)
    plt.xlabel(label)
    # plt.show()
    # 마냥 처리해주기에는 문제가 많다 
    if label in target_label:
        data.loc[data > upper] = upper
        
# columns 제거
x = x.drop(['Household_Status'],axis=1)
test_csv = test_csv.drop(['Household_Status'],axis=1)

# 스케일링
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler().fit(x)
# x = scaler.transform(x)
# test_csv = scaler.transform(test_csv)
scaler = StandardScaler().fit(x)
x = scaler.transform(x)
test_csv = scaler.transform(test_csv)

# train test split
from sklearn.model_selection import train_test_split
def load_dataset(train_size):
    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=train_size, random_state=RANDOM_STATE)
    return (x_train, y_train), (x_test, y_test)