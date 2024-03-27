from preprocessing import load_dataset, test_csv, RANDOM_STATE
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import tensorflow as tf
import random
import time

tf.random.set_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
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
        # 'tree_method' : 'hist',
        # 'device' : 'cuda',
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

def objectiveCAT(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'thread_count': 4,
        'verbose': False
    }

    model = CatBoostRegressor(**params)

    # Train the model
    model.fit(x_train, y_train, verbose=False)

    # Make predictions on the validation set
    val_preds = model.predict(x_test)

    # Calculate accuracy on the validation set
    accuracy = r2_score(y_test, val_preds)

    return accuracy

def objectiveLGBM(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': -1,
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 40),
    }

    model = LGBMRegressor(**params)

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions on the validation set
    val_preds = model.predict(x_test)

    # Calculate accuracy on the validation set
    accuracy = r2_score(y_test, val_preds)

    return accuracy

def search_param(object_fn):
    study = optuna.create_study(direction='maximize')
    study.optimize(object_fn, n_trials=300)

    best_params = study.best_params
    print("BEST PARAM: ",best_params)

    optuna.visualization.plot_param_importances(study)      # 파라미터 중요도 확인 그래프
    optuna.visualization.plot_optimization_history(study)   # 최적화 과정 시각화
    
    return best_params

xgb_params = {'n_estimators': 3879, 'max_depth': 16, 'min_child_weight': 7, 'gamma': 3, 'colsample_bytree': 1.0, 'lambda': 0.18373579449527025, 'alpha': 0.10918472960573694, 'subsample': 0.7}
cat_params = {'iterations': 459, 'learning_rate': 0.04893567915304152, 'depth': 4, 'l2_leaf_reg': 6.827813776922784, 'border_count': 184}
lgbm_params= {'n_estimators': 278, 'learning_rate': 0.011976763072611415, 'min_data_in_leaf': 32}

# model = RandomForestRegressor()
# model = XGBRegressor(**best_params)
# model = CatBoostRegressor(**best_params)
# model = LGBMRegressor(**best_params)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR

xgb = XGBRegressor(**xgb_params)
cat = CatBoostRegressor(**cat_params)
lgbm = LGBMRegressor(**lgbm_params)
rf = RandomForestRegressor()
lr = LinearRegression()
ada = AdaBoostRegressor()
svr = SVR()

def get_model(bitmask:int=0, final_estimator=None):
    model_list = [
    ('Catboost',cat),
    ('LightGBM',lgbm),
    # ('XGBoost',xgb),
    # ('RandomForest',rf),
    # ('LinearRegressor',lr),
    # ('AdaBoost',ada),
    # ('SVR',svr),
    ]
    if bitmask == 0:
        return StackingRegressor(estimators=model_list,final_estimator=final_estimator)

    estimator_models = []
    for n in range(len(model_list)):
        is_true = bitmask & (1<<n)
        if is_true:
            estimator_models.append(model_list[n])
    
    print(estimator_models)
    return StackingRegressor(estimators=estimator_models,final_estimator=final_estimator)


def columns_test(model)->None:
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
    print("============== main.py ==============")
    # columns_test(RandomForestRegressor())
    from preprocessing import submit_csv, PATH
    
    final_estimator_list = [
                            XGBRegressor(**xgb_params),
                            CatBoostRegressor(**cat_params),
                            LGBMRegressor(**lgbm_params),
                            RandomForestRegressor(),
                            AdaBoostRegressor(),
                            ]
    train_result = []
    # for n in range(0b1000): 
    for estimator in final_estimator_list:
        st = time.time()    
        model = get_model(final_estimator=estimator)
        model.fit(x_train,y_train)
        pred = model.predict(x_test)
        pred = np.where(pred < 0, 0, pred)
        y_submit = model.predict(test_csv)

        # 평가지표
        r2 = model.score(x_test,y_test)
        rmse = np.sqrt(mean_squared_error(pred,y_test))
        y_submit = np.where(y_submit < 0, 0, y_submit)
        submit_csv['Income'] = y_submit
        print(submit_csv.head(10))
        if rmse < 550:
            submit_csv.to_csv(PATH+f'submit/rmse_{rmse:4f}.csv',index=False)
        et = time.time()
        print("RandomState: ",RANDOM_STATE)
        print("R2:   ",r2)
        print("RMSE: ",rmse)
        print(f"time: {et-st:.2f}sec")
        train_result.append((f'{estimator.__class__.__name__}',rmse))
        # columns_test(model)
        
    print("Train Done")
    for d in train_result:
        print(d)
    print("next random: ",random.randint(1,100000))
# RandomState:  18947
# R2:    0.31553474445210017
# RMSE:  542.5834662845339
# time: 6.60sec

# RandomState:  13863
# ('CatBoostRegressor', 538.6564354862313)