'''
항상 좋아지는 것이 아니다 오히려 데이터 손실이 있기에 안 좋아지는 경우가 많다
그러니 0같은 데이터가 많은경우 이상치가 많은 경우 좋아질 수도 있다

보통 사용하기 전에 스케일링을 해주며 스탠다드를 쓴다 
'''
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRegressor
import sklearn as sk
import numpy as np

print(sk.__version__)   # 1.1.3

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target 
columns = datasets.feature_names
print(x.shape, y.shape) # (442, 10) (442,)

scaler = StandardScaler()
x = scaler.fit_transform(x)

origin_x = x
print('x.shape',x.shape)
for i in range(1,x.shape[1]+1):
    pca = PCA(n_components=i)
    x = pca.fit_transform(origin_x)

    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=777)

    model = RandomForestClassifier()

    model.fit(x_train,y_train)

    acc = model.score(x_test,y_test)

    print(f"{i}번째 ACC{acc}")


EVR = pca.explained_variance_ratio_
print(EVR)  # 합이 1에 가깝게 조정하는게 성능이 잘 나옴
# [4.42720256e-01 1.89711820e-01 9.39316326e-02 6.60213492e-02
#  5.49576849e-02 4.02452204e-02 2.25073371e-02 1.58872380e-02
#  1.38964937e-02 1.16897819e-02 9.79718988e-03 8.70537901e-03
#  8.04524987e-03 5.23365745e-03 3.13783217e-03 2.66209337e-03
#  1.97996793e-03 1.75395945e-03 1.64925306e-03 1.03864675e-03
#  9.99096464e-04 9.14646751e-04 8.11361259e-04 6.01833567e-04
#  5.16042379e-04 2.72587995e-04 2.30015463e-04 5.29779290e-05
#  2.49601032e-05 4.43482743e-06]

print(np.cumsum(EVR))
# [0.44272026 0.63243208 0.72636371 0.79238506 0.84734274 0.88758796
#  0.9100953  0.92598254 0.93987903 0.95156881 0.961366   0.97007138
#  0.97811663 0.98335029 0.98648812 0.98915022 0.99113018 0.99288414
#  0.9945334  0.99557204 0.99657114 0.99748579 0.99829715 0.99889898
#  0.99941502 0.99968761 0.99991763 0.99997061 0.99999557 1.        ]

import matplotlib.pyplot as plt
plt.plot(np.cumsum(EVR))
plt.grid()
plt.show()

# 1번째 ACC0.8333333333333334
# 2번째 ACC0.9473684210526315
# 3번째 ACC0.9035087719298246
# 4번째 ACC0.9298245614035088
# 5번째 ACC0.9298245614035088
# 6번째 ACC0.9298245614035088
# 7번째 ACC0.9473684210526315
# 8번째 ACC0.9298245614035088
# 9번째 ACC0.9298245614035088
# 10번째 ACC0.9385964912280702
# 11번째 ACC0.9473684210526315
# 12번째 ACC0.9298245614035088
# 13번째 ACC0.956140350877193
# 14번째 ACC0.9473684210526315
# 15번째 ACC0.9385964912280702
# 16번째 ACC0.9298245614035088
# 17번째 ACC0.9473684210526315
# 18번째 ACC0.9210526315789473
# 19번째 ACC0.9210526315789473
# 20번째 ACC0.9298245614035088
# 21번째 ACC0.9210526315789473
# 22번째 ACC0.9298245614035088
    # 23번째 ACC0.956140350877193
# 24번째 ACC0.9298245614035088
# 25번째 ACC0.9298245614035088
# 26번째 ACC0.9298245614035088
# 27번째 ACC0.9122807017543859
# 28번째 ACC0.9298245614035088
# 29번째 ACC0.9385964912280702
# 30번째 ACC0.9298245614035088