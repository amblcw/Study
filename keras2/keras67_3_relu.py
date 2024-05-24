import numpy as np
import matplotlib.pyplot as plt

# relu = lambda x : np.maximum(x,0)
relu = lambda x : [a if a >= 0 else 0 for a in x]
x = np.arange(-10,10,0.1)

plt.plot(x,relu(x))
plt.grid()
plt.show()