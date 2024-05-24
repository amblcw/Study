import numpy as np
import matplotlib.pyplot as plt

a = 1.6732632423543772848170429916717
scale=1.0507009873554804934193349852946

selu = lambda x : scale*(np.maximum(0,x)+np.minimum(0,a*(np.exp(x)-1)))

x = np.arange(-10,10,0.1)

plt.plot(x,selu(x))
plt.grid()
plt.show()
