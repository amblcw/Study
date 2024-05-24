import numpy as np
import matplotlib.pyplot as plt

mish = lambda x : x*np.tanh(np.log((1+np.exp(x))))

x = np.arange(-10,10,0.1)

plt.plot(x,mish(x))
plt.grid()
plt.show()
