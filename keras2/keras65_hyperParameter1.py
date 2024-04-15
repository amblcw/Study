import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28*28).astype(np.float32) / 255.
x_test = x_test.reshape(-1,28*28).astype(np.float32) / 255.
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# (60000, 784) (60000,) (10000, 784) (10000,)

def build_model(drop=0.05, optimizer=Adam, lr=0.0001, activation='relu', node1_output=128, node2_output=64,node3_output=32):
    inputs = Input(shape=(28*28),name='input')
    x = Dense(node1_output,activation=activation)(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2_output,activation=activation)(inputs)
    x = Dropout(drop)(x)
    x = Dense(node3_output,activation=activation)(inputs)
    x = Dropout(drop)(x)
    outputs = Dense(10,activation='softmax')(x)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer(learning_rate=lr),metrics='acc')
    return model

def create_hyperparameter():
    batchs = [100,200,300,400,500]
    optimizer = [Adam,RMSprop,Adadelta]
    dropout = [0.2,0.3,0.4,0.5]
    activation = ['relu','elu','selu','linear']
    node1 = [128,64,32,16]
    node2 = [128,64,32,16]
    node3 = [128,64,32,16]
    return {'batch_size':batchs,
            'optimizer':optimizer,
            'drop':dropout,
            'activation':activation,
            'node1_output':node1,
            'node2_output':node2,
            'node3_output':node3}
    
hyperparameters = create_hyperparameter()
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
keras_model = KerasClassifier(build_fn=build_model, verbose=1)
model = RandomizedSearchCV(keras_model,hyperparameters,cv=3,n_iter=10,verbose=1,n_jobs=16)
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

# time:  7.2 sec
# best params:     {'optimizer': <class 'keras.optimizers.optimizer_v2.adam.Adam'>, 'node3_output': 16, 'node2_output': 64, 'node1_output': 32, 'drop': 0.3, 'batch_size': 400, 'activation': 'linear'}
# best estimator:  <keras.wrappers.scikit_learn.KerasClassifier object at 0x00000144E5496760>
# best score:      0.8497833410898844
# score:           0.883400022983551
# ACC:             0.8834