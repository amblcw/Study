from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings(action='ignore')

datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape,y.shape)  #(178, 13) (178,)
# print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48

from m08_addon import m08_classifier
m08_classifier(x,y)
# r=398
# LOSS: 0.13753800094127655
# ACC:  1.0(1.0 by loss[1])

# Best result : SVC`s 1.0000

# ACC:  [0.72222222 0.72222222 0.61111111 0.62857143 0.74285714]
# 평균 ACC: 0.6854