import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28,28,1).astype(np.float32) / 255.
x_test = x_test.reshape(-1,28,28,1).astype(np.float32) / 255.
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# (60000, 784) (60000,) (10000, 784) (10000,)

def build_model(drop=0.05, optimizer=Adam, lr=0.0001, kernel_size=(2,2),activation='relu', node1_output=128, node2_output=64,node3_output=32,node4_output=256):
    inputs = Input(shape=(28,28,1),name='input')
    x = Conv2D(node1_output,kernel_size=kernel_size,activation=activation)(inputs)
    x = MaxPooling2D()(x)
    x = Dropout(drop)(x)
    x = Conv2D(node2_output,kernel_size=kernel_size,activation=activation)(inputs)
    x = MaxPooling2D()(x)
    x = Dropout(drop)(x)
    x = Conv2D(node3_output,kernel_size=kernel_size,activation=activation)(inputs)
    x = MaxPooling2D()(x)
    x = Dropout(drop)(x)
    x = Flatten()(x)
    x = Dense(node4_output,activation=activation)(x)
    outputs = Dense(10,activation='softmax')(x)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer(learning_rate=lr),metrics='acc')
    return model

def create_hyperparameter():
    batchs = [100,200,300,400,500]
    optimizer = [Adam,RMSprop,Adadelta]
    dropout = [0.01,0.05,0.1,0.15]
    activation = ['relu','elu','selu','linear']
    node1 = [128,64,32,16]
    node2 = [128,64,32,16]
    node3 = [128,64,32,16]
    node4 = [64,128,256,512]
    lr = [0.0001,0.0005,0.001,0.005]
    return {'batch_size':batchs,
            'optimizer':optimizer,
            'drop':dropout,
            'activation':activation,
            'node1_output':node1,
            'node2_output':node2,
            'node3_output':node3,
            'node4_output':node4,
            'lr':lr}
    
hyperparameters = create_hyperparameter()
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
keras_model = KerasClassifier(build_fn=build_model, verbose=1)
model = RandomizedSearchCV(keras_model,hyperparameters,cv=3,n_iter=10,verbose=1,n_jobs=4)
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

# time:  665.33 sec
# best params:     {'optimizer': <class 'keras.optimizers.optimizer_v2.rmsprop.RMSprop'>, 'node4_output': 256, 'node3_output': 16, 'node2_output': 32, 'node1_output': 32, 'lr': 0.001, 'drop': 0.05, 'batch_size': 500, 'activation': 'relu'}
# best estimator:  <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001B9E3AC1F70>
# best score:      0.9837666749954224
# score:           0.9855999946594238
# ACC:             0.9856