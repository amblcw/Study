from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import load_digits
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import numpy as np
import pickle
import joblib

x, y = load_digits(return_X_y=True)

import matplotlib.pyplot as plt
plt.yscale('symlog')
plt.boxplot(x)
# plt.show()

def fit_outlier(data):  
    data = pd.DataFrame(data)
    for label in data:
        series = data[label]
        q1 = series.quantile(0.25)      
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + iqr
        lower_bound = q1 - iqr
        
        series[series > upper_bound] = np.nan
        series[series < lower_bound] = np.nan
        print(series.isna().sum())
        series = series.interpolate()
        data[label] = series
        
    data = data.fillna(data.ffill())
    data = data.fillna(data.bfill())
    return data

x = fit_outlier(x)
# print(x.isna().sum())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8,
    # stratify=y
)

# sclaer = MinMaxScaler().fit(x_train)
sclaer = StandardScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

# pickle_path = "C:\_data\_save\_pickle_test\\"
# model = pickle.load(open(pickle_path+"m39_pickle1_save.dat",'rb'))

joblib_path = "C:\_data\_save\_joblib_test\\"
model = joblib.load(joblib_path+"m40_joblib_save.dat")

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)
print("acc score  : ",acc)

# acc score  :  0.9388888888888889 # mlogloss
# acc score  :  0.9388888888888889 동일하다



