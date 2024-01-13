'''
결측치 처리
1. 행이나 열을 삭제
2. 임의의 값을 넣는다 fillna() ffill(앞의 값으로), bfill(뒤의 값으로), median(중위값), mean(평균값)등등
3. 보간 interpolate
4. 모델로 predict
5. 부스팅 계열: 통상 결측치 이상치에 대해 자유롭다
    이걸로 결측치를 보간 할 때는 보간할 모델과 실제 훈련할 모델을 다르게 해주는 게 좋다
    같은 모델로 보간하고 훈련하면 그냥 보간없이 훈련할 때와 결과가 같다
'''
import pandas as pd
from datetime import datetime
import numpy as np

datas = ['2024-02-16','2024-02-17','2024-02-18','2024-02-19','2024-02-20','2024-02-21']
datas = pd.to_datetime(datas)
print(datas)

print("======================")
ts = pd.Series([2, np.nan, np.nan,8, 10, np.nan], index=datas)
print(ts)
# 2024-02-16     2.0
# 2024-02-17     NaN
# 2024-02-18     NaN
# 2024-02-19     8.0
# 2024-02-20    10.0
# 2024-02-21     NaN
print("======================")
ts_inter = ts.interpolate()
print(ts_inter)
# 2024-02-16     2.0
# 2024-02-17     4.0
# 2024-02-18     6.0
# 2024-02-19     8.0
# 2024-02-20    10.0
# 2024-02-21    10.0

