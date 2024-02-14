import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.DataFrame({'A': [1,2,3,4,5],
                   'B': [10,20,30,40,50],
                   'C': [5,4,3,2,1],
                   'D': [3,3,4,3,3]
                   })

print(df.corr())
