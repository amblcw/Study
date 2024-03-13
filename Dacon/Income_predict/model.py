from preprocessing import load_dataset, test_csv, RANDOM_STATE
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import tensorflow as tf
import random
import time

tf.random.set_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

from sklearn.ensemble import RandomForestRegressor, VotingClassifier
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import optuna

st = time.time()
(x_train, y_train), (x_test, y_test) = load_dataset(0.7)

# model = RandomForestRegressor()
def objectiveXGB(trial):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'max_depth' : trial.suggest_int('max_depth', 8, 16),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 300),
        'gamma' : trial.suggest_int('gamma', 1, 3),
        'learning_rate' : 0.01,
        'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
        'nthread' : -1,
        # 'tree_method' : 'gpu_hist',
        # 'predictor' : 'gpu_predictor',
        'lambda' : trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha' : trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample' : trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0]),
        'random_state' : RANDOM_STATE
    }
    
    # 학습 모델 생성
    model = XGBRegressor(**param)
    xgb_model = model.fit(x_train, y_train, verbose=True) # 학습 진행
    
    # 모델 성능 확인
    score = r2_score(xgb_model.predict(x_test), y_test)
    
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objectiveXGB, n_trials=300)

best_params = study.best_params
print("BEST PARAM: ",best_params)

optuna.visualization.plot_param_importances(study)      # 파라미터 중요도 확인 그래프
optuna.visualization.plot_optimization_history(study)   # 최적화 과정 시각화


model = XGBRegressor(**best_params)
model.fit(x_train,y_train)
pred = model.predict(x_test)
y_submit = model.predict(test_csv)

# 평가지표
r2 = model.score(x_test,y_test)
rmse = np.sqrt(mean_squared_error(pred,y_test))
et = time.time()
if __name__ == '__main__':
    print("R2:   ",r2)
    print("RMSE: ",rmse)
    print(f"time: {et-st}sec")