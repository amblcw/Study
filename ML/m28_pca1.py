'''
항상 좋아지는 것이 아니다 오히려 데이터 손실이 있기에 안 좋아지는 경우가 많다
그러니 0같은 데이터가 많은경우 이상치가 많은 경우 좋아질 수도 있다

보통 사용하기 전에 스케일링을 해주며 스탠다드를 쓴다 
'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sklearn as sk

print(sk.__version__)   # 1.1.3

datasets = load_iris()
x = datasets.data
y = datasets.target 
columns = datasets.feature_names
print(x.shape, y.shape) # (150, 4) (150,)

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=3) # n_components 개수만큼으로 줄여준다
x = pca.fit_transform(x)
print(x, x.shape, sep='\n')

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=777)

model = RandomForestClassifier()

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print(acc)

# (150, 4)
# 1.0
# (150, 3)
# 1.0
# (150, 2)
# 0.9
# (150, 1)
# 0.9333333333333333