import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
from sklearn.preprocessing import OneHotEncoder

x_data, y_data = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,train_size=0.9,stratify=y_data)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# (512, 30) (512,) (57, 30) (57,)
print(np.unique(y_train,return_counts=True))
# (array([0, 1]), array([197, 315], dtype=int64))

def build_model(drop=0.05, optimizer=Adam, lr=0.0001, activation='relu', node1_output=128, node2_output=64,node3_output=32):
    inputs = Input(shape=(30),name='input')
    x = Dense(node1_output,activation=activation)(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2_output,activation=activation)(inputs)
    x = Dropout(drop)(x)
    x = Dense(node3_output,activation=activation)(inputs)
    x = Dropout(drop)(x)
    outputs = Dense(1,activation='sigmoid')(x)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(loss='binary_crossentropy',optimizer=optimizer(learning_rate=lr),metrics='acc')
    return model

def create_hyperparameter():
    batchs = [100,200,300,400,500]
    optimizer = [Adam,RMSprop,Adadelta]
    dropout = [0.2,0.3,0.4,0.5]
    activation = ['relu','elu','selu','linear']
    node1 = [128,64,32,16]
    node2 = [128,64,32,16]
    node3 = [128,64,32,16]
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
            'callbacks':callback}
    
hyperparameters = create_hyperparameter()
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
keras_model = KerasClassifier(build_fn=build_model, verbose=1)
model = RandomizedSearchCV(keras_model,hyperparameters,cv=3,n_iter=10,verbose=1,n_jobs=12)
import time
st = time.time()
model.fit(x_train,y_train,epochs=30,verbose=2)
et = time.time()

print("time: ",round(et-st,2),"sec")
print("best params:    ",model.best_params_)
print("best estimator: ",model.best_estimator_)
print("best score:     ",model.best_score_)
print("score:          ",model.score(x_test,y_test))
from sklearn.metrics import accuracy_score
print("ACC:            ",accuracy_score(y_test,model.predict(x_test)))

# time:  31.17 sec
# best params:     {'optimizer': <class 'keras.optimizers.optimizer_v2.adam.Adam'>, 'node3_output': 128, 'node2_output': 16, 'node1_output': 32, 'drop': 0.2, 'batch_size': 100, 'activation': 'relu'}
# best estimator:  <keras.wrappers.scikit_learn.KerasClassifier object at 0x000002975433AFD0>
# best score:      0.7424836655457815
# score:           0.7719298005104065
# ACC:             0.7719298245614035