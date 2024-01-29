import numpy as np
from function_package import split_x as split2
from function_package import split_xy

a = np.array(range(1,11))
size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset)-size+1):
        subset = dataset[i: (i+size)]
        aaa.append(subset)
        
    return np.array(aaa)

bbb = split_x(a,size)
bbb2 = split2(a,size)
x, y = split_xy(a,size)
print(bbb,bbb2,sep='\n')
print(bbb.shape,bbb2.shape)

print(x,y,sep='\n')