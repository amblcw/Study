import numpy as np
import matplotlib.pyplot as plt

elu = lambda x : [a if a >=0 else (np.exp(a)-1) for a in x]

x = np.arange(-10,10,0.1)

plt.plot(x,elu(x))
plt.grid()
plt.show()
