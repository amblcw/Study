from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
import time

datasets = load_digits()
x = datasets.data   
y = datasets.target

print(x.shape, y.shape) # (1797, 64) (1797,)
print(np.unique(y,return_counts=True))  # 다중분류 확인
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64)) 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import time

def m15_classifier(x,y,param, **kwargs):
    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=333)

    model = make_pipeline(MinMaxScaler(),RandomForestRegressor(**param))
    
    st = time.time()
    model.fit(x_train,y_train)
    et = time.time()
    
    r2 = model.score(x_test, y_test)
    print("R2 : ", r2)
    print("time: ", et-st, "sec")
    
param = {'n_jobs': -1, 'min_samples_split': 3}
m15_classifier(x, y, param)

# default
# acc:  1.0
# 0.30097126960754395 sec

# GridSearchCV
# acc:  1.0
# 5.885230302810669 sec

# RandomizedSearchCV
# acc:  1.0
# 2.4038641452789307 sec
# {'n_jobs': -1, 'min_samples_split': 3}

# R2 :  0.8432405957118909
# time:  0.16321563720703125 sec