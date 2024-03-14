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

# xgb_params = {'n_estimators': 2391, 'max_depth': 16, 'min_child_weight': 19, 'gamma': 1, 'colsample_bytree': 0.8, 'lambda': 2.7858366632566747, 'alpha': 0.004919261757405025, 'subsample': 0.8}    #xgboost, always 처리

# model = RandomForestRegressor()
model = XGBRegressor(**best_params)
# model = XGBRegressor(**xgb_params)
model.fit(x_train,y_train)
pred = model.predict(x_test)
y_submit = model.predict(test_csv)

# 평가지표
r2 = model.score(x_test,y_test)
rmse = np.sqrt(mean_squared_error(pred,y_test))
et = time.time()

def columns_test():
    # print(type(model.feature_importances_))
    feature_importances_list = list(model.feature_importances_)
    feature_importances_list_sorted = sorted(feature_importances_list)
    print(feature_importances_list_sorted)
    drop_feature_idx_list = [feature_importances_list.index(feature) for feature in feature_importances_list_sorted] # 중요도가 낮은 column인덱스 부터 기재한 리스트
    print(drop_feature_idx_list)

    result_dict = {}
    for i in range(len(drop_feature_idx_list)-1): # 1바퀴에는 1개, 마지막 바퀴에는 29개 지우기, len -1해준 이유는 30개 지우면 안되니까
        drop_idx = drop_feature_idx_list[:i+1] # +1 해준 이유는 첫바퀴에 0개가 아니라 1개를 지워야하니까
        new_x_train = np.delete(x_train,drop_idx,axis=1)
        new_x_test = np.delete(x_test,drop_idx,axis=1)
        print(new_x_train.shape,new_x_test.shape)
        
        # model2 = RandomForestRegressor()
        # model2.fit(x_train,y_train)
        model2 = XGBRegressor()
        model2.set_params(early_stopping_rounds=10,**xgb_params)
        model2.fit(new_x_train,y_train,
            eval_set=[(new_x_train,y_train), (new_x_test,y_test)],
            verbose=0,
            # eval_metric='logloss',
            )
        new_result = model2.score(new_x_test,y_test)
        print(f"{i+1}개 컬럼이 삭제되었을 때 Score: ",new_result)
        result_dict[i+1] = new_result - r2    # 그대로 보면 숫자가 비슷해서 구분하기 힘들기에 얼마나 변했는지 체크
        
        
    print(result_dict)

if __name__ == '__main__':
    print("R2:   ",r2)
    print("RMSE: ",rmse)
    print(f"time: {et-st}sec")
    # print("Feacure importance: ",model.feature_importances_)
    
    