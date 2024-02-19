'''
결정트리모델은 마치 스무고개와 같아서 각 노드에서 어떤 특성의 값을 비교하여 만족하면 왼쪽, 불만족 하면 오른쪽으로 가지를 쳐서 분류합니다

각 노드에서 어떤값을 어떻게 기준으로 삼을지는 지니불순도란 지표를 활용합니다
지니 불순도란 이 노드가 얼마나 라벨이 섞여있는지를 나타냅니다
이진 분류의 경우 공식은 아래와 같습니다
지니불순도 = 1 - {(각 라벨이 차지하는 비율)^2 }의 총합
ex)만약 A, B, C 라벨이 각각 0.2, 0.3, 0.5만큼 차지한다면
지니불순도 = 1 - (0.2^2 + 0.3^2 + 0.5^2) = 0.62 입니다

그리고 트리는 부모 지니 불순도와 자식노드의 지니 불순도의 차이가 되도록 커지도록 값을 설정합니다

'''

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


x, y = load_wine(return_X_y=True)
print(x.shape, y.shape) # (178, 13) (178,)
''' x label columns
1.  Alcohol
2.  Malic acid
3.  Ash
4.  Alcalinity of ash
5.  Magnesium
6.  Total phenols
7.  Flavanoids
8.  Nonflavanoid phenols
9.  Proanthocyanins
10. Color intensity
11. Hue
12. OD280/OD315 of diluted wines
13. Proline
'''

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=333)

model = DecisionTreeClassifier(max_depth=3)

model.fit(x_train,y_train)

score = model.score(x_test,y_test)

print("score: ",score)

plt.figure(figsize=(20,15))
plot_tree(model, filled=True, feature_names=['Alcohol',
                                            'Malic_acid',
                                            'Ash',
                                            'Alcalinity_of_ash',
                                            'Magnesium',
                                            'Total_phenols',
                                            'Flavanoids',
                                            'Nonflavanoid_phenols',
                                            'Proanthocyanins',
                                            'Color_intensity',
                                            'Hue',
                                            'OD280/OD315_of_diluted wines',
                                            'Proline'
                                            ])
plt.show()

'''
결정 트리의 장점중 하나는 위와 같이 시각적으로 파악하기 쉽단 것입니다
위에 보이는 노드의 색은 짙을 수록 한 클래스가 많은 비율을 차지함을 나타냅니다

따라서 어떤 기준을 바탕으로 어떤 클래스로 분류 되었는지,
어떤 특성이 분류에 영향을 많이 미치는지 파악하기 용이합니다
'''