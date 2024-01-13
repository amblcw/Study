import pandas as pd
import numpy as np
aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])
bbb = pd.Series(aaa)
def outliers(data_out):
    q1, q2, q3 = np.percentile(data_out, [25,50,75])
    iqr = q3-q1
    lower_bound = q1 - iqr*1.5
    upper_bound = q3 + iqr*1.5
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

def outliers_pd(data_out):
    q1 = bbb.quantile(0.25)
    q3 = bbb.quantile(0.75)
    iqr = q3-q1
    lower_bound = q1 - iqr*1.5
    upper_bound = q3 + iqr*1.5
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outliers_loc = outliers(aaa)
print("이상치의 위치: ",outliers_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()