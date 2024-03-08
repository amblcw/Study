from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as np
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'

# data
np.random.seed(777)
x = 2 * np.random.rand(100, 1) -1   # -1 ~ 1 사이의 랜덤
y = 3*x**2 + 2*x + 1 + np.random.randn(100,1)   # y= 3x^2 + 2x + 1 + noise

pf = PolynomialFeatures(degree=2, include_bias=False)
x_ploy = pf.fit_transform(x)
print(x.shape,x_ploy.shape,y.shape)

# model
model1 = LinearRegression()
model2 = LinearRegression()
RF = RandomForestRegressor()

# fit
model1.fit(x,y)
model2.fit(x_ploy,y)
RF.fit(x,y)

# eval
result1 = model1.score(x,y)
result2 = model2.score(x_ploy,y)
result3 = RF.score(x,y)

print(result1,result2,result3)

# graph
plt.scatter(x,y,color='blue',label='Origin data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regressor Example')

x_plot = np.linspace(-1,1,100).reshape(-1,1)
x_plot_ploy = pf.transform(x_plot)
y_plot = model1.predict(x_plot)
y_plot2 = model2.predict(x_plot_ploy)
y_plot_rf = RF.predict(x_plot)
plt.plot(x_plot,y_plot, color='red',label='Polynomial Reg')
plt.plot(x_plot,y_plot2, color='green',label='기냥')
plt.plot(x_plot,y_plot_rf, color='black', label='RF')

plt.legend()
plt.show()
