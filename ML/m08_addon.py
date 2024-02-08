from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def m08_classifier(x,y):
    all_algorithms = all_estimators(type_filter='classifier')
    # all_algorithms = all_estimators(type_filter='regressor')

    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=333)

    N_SPLIT = 3
    kfold = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=123)

    result_list = []
    error_list = []
    for name, algorithm in all_algorithms:
        try:
            model = algorithm()
            
            print(f"======== {name} ========")
            scores = cross_val_score(model, x_train, y_train, cv=kfold)
            print("ACC: ",scores)
            print(f"평균 ACC: {round(np.mean(scores),4)}")

            y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
            acc = accuracy_score(y_test, y_predict)
            print("ACC: ", acc)
            print(f"{name:30} ACC: {acc:.4f}")
            result_list.append((name,acc))
        except Exception as e:
            print(f"{name:30} ERROR")
            error_list.append(e)
            continue
    # print('error_list: \n',error_list)
    best_result = max(result_list)[1]
    best_algirithm = result_list[result_list.index(max(result_list))][0]
    print(f'\nBest result : {best_algirithm}`s {best_result:.4f}')
    
def m08_Regressor(x,y):
    # all_algorithms = all_estimators(type_filter='classifier')
    all_algorithms = all_estimators(type_filter='regressor')

    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=333)

    N_SPLIT = 3
    kfold = KFold(n_splits=N_SPLIT, shuffle=True, random_state=123)

    result_list = []
    error_list = []
    for name, algorithm in all_algorithms:
        try:
            model = algorithm()
            
            print(f"======== {name} ========")
            scores = cross_val_score(model, x_train, y_train, cv=kfold)
            print("ACC: ",scores)
            print(f"평균 ACC: {round(np.mean(scores),4)}")

            y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
            r2 = r2_score(y_test, y_predict)
            print("r2: ", r2)
            print(f"{name:30} r2: {r2:.4f}")
            result_list.append((name,r2))
        except Exception as e:
            print(f"{name:30} ERROR")
            error_list.append(e)
            continue
    # print('error_list: \n',error_list)
    best_result = max(result_list)[1]
    best_algirithm = result_list[result_list.index(max(result_list))][0]
    print(f'\nBest result : {best_algirithm}`s {best_result:.4f}')