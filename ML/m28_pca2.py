'''
train_test_split 후 PCA처리
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

# scaler = StandardScaler()
# x = scaler.fit_transform(x)



x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=777)

scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

pca = PCA(n_components=3) # n_components 개수만큼으로 줄여준다
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
print(x_train.shape, sep='\n')

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