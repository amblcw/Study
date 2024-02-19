from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

import warnings
warnings.filterwarnings(action='ignore')
#data

x, y = load_diabetes(return_X_y=True)

print(np.unique(y,return_counts=True))