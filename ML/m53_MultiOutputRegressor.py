from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.datasets import load_linnerud
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

x, y = load_linnerud(return_X_y=True)

print(x.shape, y.shape)
# (20, 3) (20, 3)
print(x[-1],y[-1])
# [  2. 110.  43.] [138.  33.  68.]

model_list = [RandomForestRegressor(),
              LinearRegression(),
              Ridge(),
              Lasso(),
              SVR(),                # 안됨
              XGBRegressor(),
              CatBoostRegressor(),  # 안됨
              LGBMRegressor()       # 안됨
              ]

error_list = []
for model in model_list:
    try:
        model_name = model.__class__.__name__
        model.fit(x,y)
        result = model.score(x,y)
        pred = model.predict(x)
        print(result)
        # print(pred)
        print(f'{model_name}`s score: {mean_absolute_error(y,pred)}')
        print(model.predict([[2,110,43]]))
    # [[157.63  34.54  62.82]]
    except Exception as e:
        print(f"{model.__class__.__name__} error")
        error_list.append(model_name)
else:
    print(error_list)
        