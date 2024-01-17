from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import matplotlib.pyplot as plt
'''
MinMaxScaler()
최솟값을 0으로 최댓값을 1로 설정하고 그 외의 값들을 비율에 맞게 매핑해줍니다
'''
print("====MinMax====")
x = np.array([0,1,2,3,4,5]).reshape(-1,1)
minmaxscaler = MinMaxScaler()
minmaxscaler.fit(x) #x에 맞춰서 0을 0으로, 5를 1로 두고 비율 산정
x = minmaxscaler.transform(x)
print(x)
# [[0. ], [0.2], [0.4], [0.6], [0.8], [1. ]]

'''
MaxAbsScaler()
절댓값을 적용한 상태의 데이터중 제일 작은 값을 0으로, 큰값을 1로 두고 매핑을 진행합니다
매핑할때는 절댓값 적용안한 그대로 매핑하기에 범위는 -1~1이 됩니다 
위의 MinMaxScaler에서는 음수마저 양수가 되어버리는데 MaxAbsScaler는 이를 방지해줄 수 있습니다
'''
print("====MaxAbs====")
x = np.array([-5,4,3,-2,1,0]).reshape(-1,1)
maxabs = MaxAbsScaler()
maxabs.fit(x)
x = maxabs.transform(x)
print(x)
# [[-1. ], [ 0.8], [ 0.6], [-0.4], [ 0.2], [ 0. ]]

'''
StandardScaler()
데이터를 평균을 빼서 0이 중심이 되도록 이동시키고 표준편차로 나눕니다
표준편차로 나누기에 값을 0을 중심으로 모아주는 성질을 지닙니다(0에서 멀수록 나누는 값이 커지기에)
'''
print("====Standard====")
x = np.array([0,3,5,6,10]).reshape(-1,1)
standard = StandardScaler()
standard.fit(x)
x = standard.transform(x)
print(x)
# [[-1.44989302], [-0.54370988], [ 0.06041221], [ 0.36247326], [ 1.57071744]]

'''
RobustScaler()
매핑된 값 = (원본값 - 중앙값) / IQR
※IQR은 3사분면값 - 1사분면값
'''
print("====ROUBUST====")
x = np.arange(0,10)
x = np.append(x,1000000).reshape(-1,1)
print(x)
# [[      0], [      1], [      2], [      3], [      4],
#  [      5], [      6], [      7], [      8], [      9],[1000000]]
robust = RobustScaler().fit(x)
x = robust.transform(x)
print(x)
# [[-1.00000e+00], [-8.00000e-01], [-6.00000e-01], [-4.00000e-01], [-2.00000e-01],
#  [ 0.00000e+00], [ 2.00000e-01], [ 4.00000e-01], [ 6.00000e-01], [ 8.00000e-01], [ 1.99999e+05]]
'''
여기서 중앙값은 5이며 3사분면 값은 7, 1사분면값은 2가 됩니다 따라서 IQR = 7-2 = 5입니다
따라서 
0의 경우 (0-5)/5 = -1로 스케일됩니다
5의 경우 (5-5)/5 = 0으로 스케일됩니다
1000000의 경우 (1000000-5)/5 = 199999로 스케일됩니다 

보다시피 RobustScaler는 평균도 분산도 사용하지 않기에 이상치에 영향을 제일 적게받습니다
따라서 이상치가 우려되는 데이터에 사용되기 적합합니다
'''