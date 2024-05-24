import numpy as np
import matplotlib.pyplot as plt

sigmoid = lambda x : 1/(1+np.exp(-x))
silu = lambda x : x*sigmoid(x)

x = np.arange(-10,10,0.1)

plt.plot(x,silu(x))
plt.grid()
plt.show()
