import numpy as np

x = np.array([1,2,3,4,5])

mean = x.sum() / len(x)
deviation = [a-mean for a in x]
variance = [a*a for a in deviation]
standard_deviation = [np.sqrt(a) for a in variance]

print("mean:",mean)
print("deviation:",deviation)
print("variance:", variance)
print("standard deviation:",standard_deviation)
'''
mean: 3.0
deviation: [-2.0, -1.0, 0.0, 1.0, 2.0]
variance: [4.0, 1.0, 0.0, 1.0, 4.0]
standard deviation: [2.0, 1.0, 0.0, 1.0, 2.0]
'''