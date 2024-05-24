import numpy as np
import matplotlib.pyplot as plt

tanh = lambda x : (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

x = np.arange(-10,10,0.1)

plt.plot(x,tanh(x))
plt.grid()
plt.show()