from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping

for i in range(10):
    print(i)