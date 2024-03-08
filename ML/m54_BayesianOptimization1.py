import pandas as pd
import numpy as np

param_bounds = {
    'x1' : (-1,5),
    'x2' : (0,4),
}

def y_funstion(x1,x2):
    return -x1**2 - (x2-2)**2 + 10

from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(f= y_funstion,
                                 pbounds=param_bounds,
                                 random_state=47,
                                 )

optimizer.maximize(init_points=5,
                   n_iter=1000,
                   )

print(optimizer.max)