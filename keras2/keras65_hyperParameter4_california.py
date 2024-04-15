import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
from sklearn.preprocessing import OneHotEncoder

x_data, y_data = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,train_size=0.9)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# (18576, 8) (18576,) (2064, 8) (2064,)
print(np.unique(y_train,return_counts=True))
# (array([0.14999, 0.175  , 0.225  , ..., 4.991  , 5.     , 5.00001]), array([  4,   1,   4, ...,   1,  24, 870], dtype=int64))

def build_model(drop=0.05, optimizer=Adam, lr=0.0001, activation='relu', node1_output=128, node2_output=64,node3_output=32):
    inputs = Input(shape=(8),name='input')
    x = Dense(node1_output,activation=activation)(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2_output,activation=activation)(inputs)
    x = Dropout(drop)(x)
    x = Dense(node3_output,activation=activation)(inputs)
    x = Dropout(drop)(x)
    outputs = Dense(1,activation='linear')(x)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(loss='mse',optimizer=optimizer(learning_rate=lr))
    return model

def create_hyperparameter():
    batchs = [100,200,300,400,500]
    optimizer = [Adam,RMSprop,Adadelta]
    dropout = [0.01,0.05,0.1,0.15]
    activation = ['relu','elu','selu','linear']
    node1 = [64,32,16]
    node2 = [64,32,16]
    node3 = [64,32,16]
    lr = [0.001,0.005,0.01,0.05]
    
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    es = EarlyStopping(monitor='loss',patience=5,mode='auto',restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='loss',patience=3,mode='auto',factor=0.7)
    callback = [es,rlr]
    
    return {'batch_size':batchs,
            'optimizer':optimizer,
            'drop':dropout,
            'activation':activation,
            'node1_output':node1,
            'node2_output':node2,
            'node3_output':node3,
            'lr':lr,
            'callbacks':callback}
    
hyperparameters = create_hyperparameter()
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
keras_model = KerasRegressor(build_fn=build_model, verbose=1)
model = RandomizedSearchCV(keras_model,hyperparameters,cv=3,n_iter=30,verbose=1,n_jobs=10)
import time
st = time.time()
model.fit(x_train,y_train,epochs=30,verbose=2)
et = time.time()

print("time: ",round(et-st,2),"sec",sep='')
print("best params:    ",model.best_params_)
print("best estimator: ",model.best_estimator_)
print("best score:     ",model.best_score_)
print("score:          ",model.score(x_test,y_test))
from sklearn.metrics import accuracy_score, r2_score
print("R2:             ",r2_score(y_test,model.predict(x_test)))

# r2=0.672248681511576 keras1
# time: 429.13sec
# best params:     {'optimizer': <class 'keras.optimizers.optimizer_v2.adam.Adam'>, 'node3_output': 32, 'node2_output': 32, 'node1_output': 64, 'lr': 0.05, 'drop': 0.01, 'callbacks': <keras.callbacks.ReduceLROnPlateau object at 0x0000014441EBF970>, 'batch_size': 100, 'activation': 'selu'}
# best estimator:  <keras.wrappers.scikit_learn.KerasRegressor object at 0x000001444B9C4A00>
# best score:      -0.6550900737444559
# score:           -0.5398083925247192
# R2:              0.5887101844929682