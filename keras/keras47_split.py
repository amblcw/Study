import numpy as np
from function_package import split_x as split2
from function_package import split_xy

a = np.array(range(1,11))
size = 5

def split_x(dataset, size):             # dataset 과 size를 인자로 받는다, dataset은 리스트이며 size는 정수여야한다
    aaa = []                            # 반환하기 위한 임시 리스트 
    for i in range(len(dataset)-size+1):# dataset을 size만큼 자르면 len(dataset) - size + 1 만큼 자를 수있다
        subset = dataset[i: (i+size)]   # dataset을 i번 부터 size만큼 자른다, i 는 0부터 len(dataset)-size 까지의 정수이다
        aaa.append(subset)              # 잘라낸 데이터를 aaa에 붙인다
        
    return np.array(aaa)                # 완성된 리스트를 반환한다

bbb = split_x(a,size)
bbb2 = split2(a,size)
x, y = split_xy(a,size)
print(bbb,bbb2,sep='\n')
print(bbb.shape,bbb2.shape)

print(x,y,sep='\n')