from preprocessing import submit_csv, PATH
from model import y_submit, r2, rmse
import pandas as pd

print("============== main.py ==============")

submit_csv['Income'] = y_submit
print(submit_csv.head(10))

submit_csv.to_csv(PATH+f'submit/rmse_{rmse:4f}.csv',index=False)

print("R2:   ",r2)
print("RMSE: ",rmse)