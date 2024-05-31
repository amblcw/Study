import numpy as np

# 예제 데이터를 생성하는 함수 (각 루프마다 다른 크기의 배열 생성)
def generate_array(i):
    return np.random.rand(3, 4) * i

# 배열을 저장할 리스트 초기화
arrays = []

# for 루프를 통해 배열 생성 및 리스트에 추가
for i in range(5):
    array = generate_array(i)
    print(array.shape)
    arrays.append(array)

# 리스트에 있는 모든 배열을 새로운 축을 따라 쌓음
combined_array = np.vstack(arrays)

print(combined_array)
print(combined_array.shape)
