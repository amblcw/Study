import numpy as np
import matplotlib.pyplot as plt

reaky_relu = lambda x : np.maximum(x, 0.01*x)

x = np.arange(-10,10,0.1)

plt.plot(x,reaky_relu(x))
plt.grid()
plt.show()
