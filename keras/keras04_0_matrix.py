import numpy as np

x1 = np.array([1,2,3])
x2 = np.array([[1,2,3]])
x3 = np.array([[1,2],[3,4]])
x4 = np.array([[1,2],[3,4],[5,6]])
x5 = np.array([[[1,2],[3,4],[5,6]]])
x6 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
x7 = np.array([[[[1,2,3,4,5],[6,7,8,9,10]]]])
x8 = np.array([[1,2,3],[4,5,6]])
x9 = np.array([[[[1]]],[[[2]]]])

print(f"{x1.shape=}",f"{x2.shape=}",f"{x3.shape=}",f"{x4.shape=}",f"{x5.shape=}",
      f"{x6.shape=}",f"{x7.shape=}",f"{x8.shape=}",f"{x9.shape=}", sep='\n')

# x1.shape=(3,)
# x2.shape=(1, 3)
# x3.shape=(2, 2)
# x4.shape=(3, 2)
# x5.shape=(1, 3, 2)
# x6.shape=(2, 2, 2)
# x7.shape=(1, 1, 2, 5)
# x8.shape=(2, 3)
# x9.shape=(2, 1, 1, 1)