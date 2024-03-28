from preprocessing import submit_csv, PATH
from model import y_submit, r2, rmse, if_main
import pandas as pd
import time

print("============== main.py ==============")
st = time.time()
submit_csv['Income'] = y_submit
print(submit_csv.head(10))

submit_csv.to_csv(PATH+f'submit/rmse_{rmse:4f}.csv',index=False)
et = time.time()
print("R2:   ",r2)
print("RMSE: ",rmse)
print(f"time: {et-st:.2f}sec")