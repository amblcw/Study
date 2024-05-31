# https://dongsarchive.tistory.com/74 그림 참고하면 이해가 더 쉬움
import torch
from torch.nn import *

x = torch.tensor([
    [[[1.,2.,3.],
     [-1.,2.,2.],
     [3.,2.,4.]], # 평균 2
    [[-1.,-1.,-1.],
     [0.,0.,0.],
     [1.,1.,1.]]],# 평균 0
    [[[0.,1.,2.],
     [-2.,1.,1.],
     [2.,1.,3.]], # 평균 1
    [[-1.,-1.,-1.],
     [0.,0.,0.],
     [1.,1.,1.]]] # 평균 0
])
print(x.shape)
# (2,2,3,3) 
layer_norm = LayerNorm([2,3,3])#,elementwise_affine=False,bias=False)
batch_norm = BatchNorm2d(2,affine=False)
instance_norm = InstanceNorm2d(2)

print("BatchNorm\n",batch_norm(x))
print("LayerNorm\n",layer_norm(x))
print("InstanceNorm\n",instance_norm(x))
'''
BatchNorm
배치사이즈 내의 동일 채널(여기선 1,3행렬과 2,4행렬)간의 표준화
 tensor([[[[-0.3511,  0.3511,  1.0534],
          [-1.7556,  0.3511,  0.3511],
          [ 1.0534,  0.3511,  1.7556]],

         [[-1.2247, -1.2247, -1.2247],
          [ 0.0000,  0.0000,  0.0000],
          [ 1.2247,  1.2247,  1.2247]]],


        [[[-1.0534, -0.3511,  0.3511],
          [-2.4579, -0.3511, -0.3511],
          [ 0.3511, -0.3511,  1.0534]],

         [[-1.2247, -1.2247, -1.2247],
          [ 0.0000,  0.0000,  0.0000],
          [ 1.2247,  1.2247,  1.2247]]]])


LayerNorm
한 이미지 마다 표준화(여기선 1,2행렬과 3,4행렬)
 tensor([[[[ 0.0000,  0.6708,  1.3416],
          [-1.3416,  0.6708,  0.6708],
          [ 1.3416,  0.6708,  2.0125]],

         [[-1.3416, -1.3416, -1.3416],
          [-0.6708, -0.6708, -0.6708],
          [ 0.0000,  0.0000,  0.0000]]],


        [[[-0.4121,  0.4121,  1.2362],
          [-2.0604,  0.4121,  0.4121],
          [ 1.2362,  0.4121,  2.0604]],

         [[-1.2362, -1.2362, -1.2362],
          [-0.4121, -0.4121, -0.4121],
          [ 0.4121,  0.4121,  0.4121]]]], grad_fn=<NativeLayerNormBackward0>)
          
          
InstanceNorm
각 채널마다 표준화(1,2,3,4행렬 각각)
 tensor([[[[-0.7500,  0.0000,  0.7500],
          [-2.2500,  0.0000,  0.0000],
          [ 0.7500,  0.0000,  1.5000]],

         [[-1.2247, -1.2247, -1.2247],
          [ 0.0000,  0.0000,  0.0000],
          [ 1.2247,  1.2247,  1.2247]]],


        [[[-0.7500,  0.0000,  0.7500],
          [-2.2500,  0.0000,  0.0000],
          [ 0.7500,  0.0000,  1.5000]],

         [[-1.2247, -1.2247, -1.2247],
          [ 0.0000,  0.0000,  0.0000],
          [ 1.2247,  1.2247,  1.2247]]]])
'''