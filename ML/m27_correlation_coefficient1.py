import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import matplotlib

print(sklearn.__version__)

# data
datasets = load_iris()
x = datasets.data
y = datasets.target

df = pd.DataFrame(x, columns=datasets.feature_names)
df['target(y)'] = y
print(df.head)

print("======= 상관계수 히트맵 =======")
print(df.corr())

import seaborn as sns 
plt.figure(figsize=(10,8))
sns.set(style='white')
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()

print(matplotlib.__version__)  # 3.7.2 | 3.8.0에선 제대로 나오지 않음

""" 
def plot_FI(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)
    plt.title(type(model).__name__)
    plt.show()

print(np.unique(y,return_counts=True))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123, stratify=y)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

param = {'random_state':123}
model_list = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for i, model in enumerate(model_list):
    model.fit(x_train,y_train)

    acc = model.score(x_test,y_test)
    print(type(model).__name__,"`s ACC: ",acc,sep='')
    print(type(model).__name__, ":",model.feature_importances_, "\n")
    # plot_FI(model)
    
from xgboost.plotting import plot_importance
    
plot_importance(model_list[-1])
plt.show() """


