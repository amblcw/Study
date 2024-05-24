import numpy as np
import matplotlib.pyplot as plt

# softmax = lambda x : [np.exp(a)/ for a in x]

def softmax(x:list)->list:
    return [a/sum(np.exp(x)) for a in np.exp(x)]

x = np.arange(1,5,1)
print(sum(softmax(x)))

# plt.bar(x, softmax(x))
plt.pie(softmax(x),softmax(x),shadow=True,startangle=90)
plt.grid()
plt.show()
