'''
1. 공분산, 공분산행렬, 상관계수
공분산은 두 집단간의 상관관계를 표현한 수치입니다
두 집단을 X, Y라고 할 때
Cov(X,Y) = E[(x-mean(x))*(y-mean(y))] = X와Y요소들의 곱의 평균 - (X의 평균 * Y의 평균) 입니다

두 집단이 서로 비례한다면 공분산은 양의 값을
두 집단이 서로 반비례하면 공분산은 음의 값을
두 집단이 서로 연관이 없다면 공분산은 0을 가집니다
'''
import numpy as np
x = [1,2,3,4,5]
y = [5,4,3,2,1]

mean_x = np.mean(x)
mean_y = np.mean(y)

cov = np.mean([(a-mean_x)*(b-mean_y) for a,b in zip(x,y)])
print(cov, np.cov(x,y,ddof=0)[0,1])
# -2.0 -2.0
'''
공분산 행렬은 둘 이상의 집단간의 공분산을 모은 행렬입니다
위의 경우에도 공분산 행렬로 나타낸다면
[ X와 X의 공분산, X와 Y의 공분산
  Y와 X의 공분산, Y와 Y의 공분산]
'''
covXX = np.mean([(a-mean_x)*(a-mean_x) for a in x])
covXY = np.mean([(a-mean_x)*(b-mean_y) for a,b in zip(x,y)])
covYX = np.mean([(b-mean_y)*(a-mean_x) for a,b in zip(x,y)])
covYY = np.mean([(b-mean_y)*(b-mean_y) for b in y])
print([[covXX,covXY],[covYX,covYY]])
print(np.cov(x,y,ddof=0))
#[[2.0, -2.0], [-2.0, 2.0]]
#[[ 2. -2.]
# [-2.  2.]]
 
'''
공분산은 절대적인 상관관계를 나타내지 않습니다 공분산은 데이터의 scale에 영향을 받습니다
'''
x = [10,20,30,40,50]
y = [50,40,30,20,10]

mean_x = np.mean(x)
mean_y = np.mean(y)

cov = np.mean([(a-mean_x)*(b-mean_y) for a,b in zip(x,y)])
print(cov, np.cov(x,y,ddof=0)[0,1])
'''
-200.0 -200.0

따라서 절대적인 상관관계값을 얻기위해 상관계수는 공분산에서 (X의분산 * Y의 분산)의 제곱근 으로 나눕니다
따라서 상관계수는 -1에서 1 사이의 값을 가집니다
'''
correlation = covXY / np.sqrt(covXX*covYY) # 자기자신과의 공분산은 자신의 분산과 같다
print(correlation, np.corrcoef(x,y)[0,1])
# -1.0 -1.0