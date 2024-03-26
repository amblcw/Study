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
                            # XGBRegressor(**xgb_params),
                            CatBoostRegressor(**cat_params),
                            # LGBMRegressor(**lgbm_params),
                            # RandomForestRegressor(),
                            # AdaBoostRegressor(),
                            ]
    train_result = []
    # for n in range(0b1000): 
    for estimator in final_estimator_list:
        st = time.time()    
        model = get_model(final_estimator=estimator)
        model.fit(x_train,y_train)
        pred = model.predict(x_test)
        y_submit = model.predict(test_csv)

        # 평가지표
        r2 = model.score(x_test,y_test)
        rmse = np.sqrt(mean_squared_error(pred,y_test))
        submit_csv['Income'] = y_submit
        print(submit_csv.head(10))

        submit_csv.to_csv(PATH+f'submit/rmse_{rmse:4f}.csv',index=False)
        et = time.time()
        print("R2:   ",r2)
        print("RMSE: ",rmse)
        print(f"time: {et-st:.2f}sec")
        train_result.append((f'{estimator.__class__.__name__}',rmse))
        
        columns_test(model)
        
    print("Train Done")
    for d in train_result:
        print(d)
# Train Done
# ('0', 578.3871586857902)
# ('1', 570.7883530694928)
# ('10', 569.0854186584655)
    # ('11', 568.4101040198864) BEST
# ('100', 727.6720542960642)
# ('101', 610.4365578503008)
# ('110', 611.4097463391362)
# ('111', 586.8006177938422)
# ('1000', 606.0101374198864)
# ('1001', 576.0125159644905)
# ('1010', 577.7139351056754)
# ('1011', 571.921530547641)
# ('1100', 638.756242990304)
# ('1101', 598.6094683203111)
# ('1110', 600.2821656474449)
# ('1111', 585.2763573422645)
# ('10000', 601.6757595706806)
# ('10001', 579.7191559984883)
# ('10010', 577.7495489581904)
# ('10011', 573.5042645036777)
# ('10100', 615.5549888308218)
# ('10101', 590.0226416377549)
# ('10110', 589.7945877892554)
# ('10111', 580.4594199563128)
# ('11000', 583.8821641142752)
# ('11001', 575.8796345215703)
# ('11010', 574.9521815279179)
# ('11011', 572.6576227622335)
# ('11100', 600.7177592915224)
# ('11101', 586.9138284786252)
# ('11110', 586.6475975288766)
# ('11111', 580.1992611822428)
# ('100000', 645.5827756558655)
# ('100001', 597.2176891275202)
# ('100010', 592.6836425884101)
# ('100011', 583.1125631149276)
# ('100100', 630.6291453174282)
# ('100101', 598.6573495877795)
# ('100110', 637.7425663873828)
# ('100111', 584.9586514703569)
# ('101000', 598.3797465833243)
# ('101001', 586.4591936492682)
# ('101010', 585.697088668466)
# ('101011', 578.4985955080855)
# ('101100', 612.0625663189261)
# ('101101', 616.6317221021882)
# ('101110', 593.3486159286628)
# ('101111', 583.722682617836)
# ('110000', 606.7655760972718)
# ('110001', 589.418387495343)
# ('110010', 586.717270275451)
# ('110011', 580.4283482583704)
# ('110100', 603.8035205112378)
# ('110101', 588.1587439620264)
# ('110110', 596.5357491993535)
# ('110111', 581.6057284574441)
# ('111000', 590.4733963480604)
# ('111001', 581.6259603774485)
# ('111010', 581.4029702981638)
# ('111011', 578.2980385868306)
# ('111100', 596.4677664755949)
# ('111101', 585.4951251458522)
# ('111110', 586.2745080157715)
# ('111111', 590.8961913300051)
# ('1000000', 619.7924039712204)
# ('1000001', 582.1535424673434)
# ('1000010', 581.244582492533)
# ('1000011', 574.0259431220554)
# ('1000100', 617.3225332389293)
# ('1000101', 589.8250217201006)
# ('1000110', 590.0573940263193)
# ('1000111', 580.0272487096105)
# ('1001000', 585.4424857516673)
# ('1001001', 575.0067344017774)
# ('1001010', 575.0941672743666)
# ('1001011', 572.0000291729202)
# ('1001100', 600.2814961323695)
# ('1001101', 586.2154199279918)
# ('1001110', 585.8642614543588)
# ('1001111', 579.0542573749367)
# ('1010000', 603.390639304304)
# ('1010001', 585.403332382476)
# ('1010010', 584.3197109068528)
# ('1010011', 577.9014215432248)
# ('1010100', 598.2045645464212)
# ('1010101', 585.4387249701692)
# ('1010110', 585.1872379485986)
# ('1010111', 579.1368054172444)
# ('1011000', 584.9143949713169)
# ('1011001', 578.3820745400949)
# ('1011010', 577.8899744634635)
# ('1011011', 573.9761122570226)
# ('1011100', 590.592604667542)
# ('1011101', 582.5204860867683)
# ('1011110', 582.2635986705662)
# ('1011111', 577.8787776518478)
# ('1100000', 597.4572963095046)
# ('1100001', 585.7210604128887)
# ('1100010', 585.9271924211702)
# ('1100011', 579.6943342250007)
# ('1100100', 600.5912592715322)
# ('1100101', 588.018046857281)
# ('1100110', 585.7235085979662)
# ('1100111', 579.7560045420372)
# ('1101000', 584.1422522367569)
# ('1101001', 578.4079663548754)
# ('1101010', 578.2463260265173)
# ('1101011', 574.927273730431)
# ('1101100', 600.9528032087386)
# ('1101101', 590.7147303954598)
# ('1101110', 583.1044675166189)
# ('1101111', 578.7673353044651)
# ('1110000', 597.4958586268922)
# ('1110001', 585.1246980373669)
# ('1110010', 586.0411706552525)
# ('1110011', 583.4815709042857)
# ('1110100', 590.1038322298419)
# ('1110101', 585.4602688011895)
# ('1110110', 583.4314479843771)
# ('1110111', 579.8256726269163)
# ('1111000', 586.2251303962126)
# ('1111001', 580.0154963791101)
# ('1111010', 579.5595779411567)
# ('1111011', 588.3101344206128)
# ('1111100', 586.9329026341804)
# ('1111101', 582.0742844011387)
# ('1111110', 582.0185825334066)
# ('1111111', 578.6715121352336)