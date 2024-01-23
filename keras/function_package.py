from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np

def image_scaler(x_train:np.array, x_test:np.array, scaler:str):
    '''
    image를 scaling하는 함수입니다
    반환값은 x_train, x_test입니다
    scaler 값은 minmax, standard, robust중 하나로 해주세요
    '''
    xtr0, xtr1, xtr2, xtr3 = (0, 0, 0, 0)
    if(len(x_train.shape)==3):
        xtr0, xtr1, xtr2, xtr3 = (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        xt0, xt1, xt2, xt3 = (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    else:
        xtr0, xtr1, xtr2, xtr3 = (x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])
        xt0, xt1, xt2, xt3 = (x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])

    x_train = x_train.reshape(xtr0, xtr1*xtr2*xtr3)
    x_test = x_test.reshape(xt0, xt1*xt2*xt3)

    if scaler == 'minmax':
        minmax = MinMaxScaler().fit(x_train)
        x_train = minmax.transform(x_train)
        x_test = minmax.transform(x_test)
    elif scaler == 'standard':
        standard = StandardScaler().fit(x_train)
        x_train = standard.transform(x_train)
        x_test = standard.transform(x_test)
    elif scaler == 'robust':
        robust = RobustScaler().fit(x_train)
        x_train = robust.transform(x_train)
        x_test = robust.transform(x_test)
    else:
        print(f"set wrong scaler({scaler}), set by 'minmax' or 'stardard' or 'robust'.")

    x_train = x_train.reshape(xtr0, xtr1, xtr2, xtr3)
    x_test = x_test.reshape(xt0, xt1, xt2, xt3)
    
    return x_train, x_test