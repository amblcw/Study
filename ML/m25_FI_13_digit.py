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


''' 25퍼 미만 열 삭제 '''
columns = datasets.feature_names
# columns = x.columns
x = pd.DataFrame(x,columns=columns)
print("x.shape",x.shape)
''' 이 밑에 숫자에 얻은 feature_importances 넣고 줄 끝마다 \만 붙여주기'''
fi_str = "0.         0.00077333 0.0026385  0.01104499 0.00600033 0.04942167\
 0.         0.         0.         0.00380811 0.00412737 0.00144355\
 0.00796947 0.01158992 0.00521998 0.00103111 0.         0.00636011\
 0.00521263 0.01073529 0.04922624 0.09467774 0.00340266 0.\
 0.00153333 0.00077333 0.07689658 0.0610088  0.0518341  0.01383698\
 0.0143471  0.         0.         0.0555721  0.0097232  0.00483742\
 0.07193319 0.01344586 0.01620286 0.         0.         0.00408443\
 0.07616199 0.05744186 0.02086224 0.0117224  0.01384963 0.\
 0.         0.         0.01103088 0.         0.00429678 0.01028292\
 0.02135063 0.         0.         0.         0.01913081 0.00537631\
 0.06711274 0.001392   0.00927655 0.        "
 
''' str에서 숫자로 변환하는 구간 '''
fi_str = fi_str.split()
fi_float = [float(s) for s in fi_str]
print(fi_float)
fi_list = pd.Series(fi_float)

''' 25퍼 미만 인덱스 구하기 '''
low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
print('low_idx_list',low_idx_list)

''' 25퍼 미만 제거하기 '''
low_col_list = [x.columns[index] for index in low_idx_list]
# 이건 혹여 중복되는 값들이 많아 25퍼이상으로 넘어갈시 25퍼로 자르기
if len(low_col_list) > len(x.columns) * 0.25:   
    low_col_list = low_col_list[:int(len(x.columns)*0.25)]
print('low_col_list',low_col_list)
x.drop(low_col_list,axis=1,inplace=True)
print("after x.shape",x.shape)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=123)
param = [{'random_state': np.random.randint(1000,size=100)}]
model_list = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for model in model_list:
    model = RandomizedSearchCV(model,param,cv=5,n_iter=10)
    model.fit(x_train,y_train)

    acc = model.score(x_test,y_test)
    print(type(model).__name__,"`s ACC: ",acc,sep='')
    print('best_param:' , model.best_params_)
    # print(type(model).__name__, ":",model.feature_importances_, "\n")

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
""" 
DecisionTreeClassifier`s ACC: 0.8194444444444444
DecisionTreeClassifier : [0.         0.00077333 0.0026385  0.01104499 0.00600033 0.04942167
 0.         0.         0.         0.00380811 0.00412737 0.00144355
 0.00796947 0.01158992 0.00521998 0.00103111 0.         0.00636011
 0.00521263 0.01073529 0.04922624 0.09467774 0.00340266 0.
 0.00153333 0.00077333 0.07689658 0.0610088  0.0518341  0.01383698
 0.0143471  0.         0.         0.0555721  0.0097232  0.00483742
 0.07193319 0.01344586 0.01620286 0.         0.         0.00408443
 0.07616199 0.05744186 0.02086224 0.0117224  0.01384963 0.
 0.         0.         0.01103088 0.         0.00429678 0.01028292
 0.02135063 0.         0.         0.         0.01913081 0.00537631
 0.06711274 0.001392   0.00927655 0.        ]

RandomForestClassifier`s ACC: 0.9777777777777777
RandomForestClassifier : [0.00000000e+00 2.26787060e-03 1.70988253e-02 1.05719484e-02
 9.71122179e-03 1.96361159e-02 7.52990286e-03 6.63317516e-04
 9.25128309e-05 1.01665392e-02 2.51681513e-02 7.87220648e-03
 1.62070997e-02 2.89322209e-02 5.19024925e-03 5.40732498e-04
 6.84209419e-05 9.46808460e-03 1.99902630e-02 2.40670140e-02
 2.99267182e-02 5.46467980e-02 7.30666096e-03 3.57047775e-04
 6.64651831e-05 1.66579334e-02 4.15356516e-02 2.41853531e-02
 3.06614363e-02 2.77736563e-02 2.72288739e-02 2.79782179e-05
 0.00000000e+00 3.01599289e-02 2.51601540e-02 2.10040285e-02
 3.81224068e-02 1.87254205e-02 2.63208382e-02 0.00000000e+00
 0.00000000e+00 1.14864125e-02 3.45858505e-02 4.28514473e-02
 2.00439374e-02 1.62217590e-02 2.08143980e-02 8.04151682e-05
 3.86928793e-05 2.15700106e-03 1.76239627e-02 2.59422085e-02
 1.46802191e-02 2.42813144e-02 2.47269138e-02 1.71801075e-03
 2.48932679e-05 2.55430489e-03 2.41716989e-02 1.12087272e-02
 2.41733904e-02 2.46484014e-02 1.73106992e-02 3.54529470e-03]

GradientBoostingClassifier`s ACC: 0.9583333333333334
GradientBoostingClassifier : [0.00000000e+00 8.07128891e-04 1.07629881e-02 3.51615233e-03
 2.19257643e-03 5.92487382e-02 4.32011264e-03 7.13666238e-04
 6.56467229e-04 2.30845164e-03 2.18271097e-02 1.20997268e-03
 7.44909279e-03 7.64448720e-03 2.50059015e-03 1.12786468e-03
 2.00127157e-04 2.05616151e-03 1.38473709e-02 3.74507383e-02
 2.25508483e-02 8.99859875e-02 3.70181095e-03 6.02729101e-06
 1.40235950e-05 2.99397600e-03 4.34776398e-02 1.46043726e-02
 3.20837466e-02 2.69340730e-02 1.05922468e-02 6.11083132e-04
 0.00000000e+00 6.51016156e-02 2.70596654e-03 6.77948525e-03
 7.03300766e-02 1.11707347e-02 1.22588666e-02 0.00000000e+00
 0.00000000e+00 5.17018101e-03 7.94548379e-02 7.66219352e-02
 1.16309328e-02 2.03203918e-02 3.54450256e-02 2.90278537e-05
 4.56840004e-04 1.43245434e-03 7.19831175e-03 1.16810286e-02
 9.38780630e-03 1.33999355e-02 2.62953444e-02 1.39007973e-03
 2.72030836e-04 7.01253882e-05 9.70095855e-03 1.72851650e-03
 5.90263462e-02 6.88144219e-03 2.18869232e-02 4.77714665e-03]

XGBClassifier`s ACC: 0.9638888888888889
XGBClassifier : [0.         0.03191047 0.01254289 0.00459743 0.00548403 0.03840715
 0.00522263 0.01285057 0.         0.00957167 0.01437867 0.00560509
 0.00831374 0.00959539 0.00293379 0.01588055 0.         0.00752004
 0.00580061 0.03555962 0.01018464 0.04830588 0.00428059 0.
 0.         0.00730744 0.03246765 0.00803092 0.04202451 0.01813976
 0.0230799  0.         0.         0.06601543 0.00546573 0.00894697
 0.05283065 0.01979204 0.02676693 0.         0.         0.01150967
 0.04082786 0.04033075 0.01122894 0.00886908 0.03779897 0.
 0.         0.00463531 0.00508138 0.01397322 0.00371399 0.01375694
 0.03390912 0.0098333  0.         0.00251326 0.00985752 0.0090441
 0.06077936 0.01497968 0.03824107 0.02930311] """
 
#  RandomizedSearchCV`s ACC: 0.8166666666666667
# best_param: {'random_state': 267}
# RandomizedSearchCV`s ACC: 0.9805555555555555
# best_param: {'random_state': 902}
# RandomizedSearchCV`s ACC: 0.9555555555555556
# best_param: {'random_state': 160}
# RandomizedSearchCV`s ACC: 0.9638888888888889
# best_param: {'random_state': 515}

# after
# RandomizedSearchCV`s ACC: 0.8222222222222222
# best_param: {'random_state': 634}
# RandomizedSearchCV`s ACC: 0.9805555555555555
# best_param: {'random_state': 287}
# RandomizedSearchCV`s ACC: 0.9666666666666667
# best_param: {'random_state': 101}
# RandomizedSearchCV`s ACC: 0.9555555555555556
# best_param: {'random_state': 149}