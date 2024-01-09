# list 예시
a = [1,2,3,4]
b = [1, 'a', [1,2,3,4]] #다른 자료형이 공존하거나 리스트 안에 리스트도 가능

# 리스트의 가장 큰 특징은 변경이 자유롭단 것입니다
c = a+b     #두 리스트를 붙일 수도 있습니다
print(c)    #[1, 2, 3, 4, 1, 'a', [1, 2, 3, 4]]

a.append(5) #이렇게 뒤에 붙이는 것 또한 자유롭습니다
print(a)    #[1, 2, 3, 4, 5]

# 접근은 인덱스 번호를 통해 접근하며 첫 번호가 0번임에 주의합니다
print(a[0],a[1]) # 1 2

######################################################
# tuple 예시
c = (1,'a',[1,2],4) #다양한 자료형 가능, 내부에 다른 리스트나 튜플등도 가능
# 튜플의 가장 큰 특징은 변경 불가능입니다 
# c[1] = 2
# 위 코드 실행시 아래와 같이 오류가 뜹니다
# c[1] = 2
#     ~^^^
# TypeError: 'tuple' object does not support item assignment

# 튜플의 접근 또한 인덱스 번호를 통해 이루어지며 첫 번호는 0입니다
print(c[0], c[1]) # 1 a

#######################################################
# dictionary 예시
d = {'first': 1, 'second': "this is second", 'inner_dic':{'in_a':1,'in_b':2}} #여러 자료형가능, 내부에 리스트, 튜플, 딕셔너리등등 가능
for key , value in d.items(): #딕셔너리의 함수 items()는 (key, value)모양의 튜플로 반환해줍니다
    print(f"{key}: {value},")
# first: 1,
# second: this is second,
# inner_dic: {'in_a': 1, 'in_b': 2},

# 딕셔너리의 가장 큰 특징은 항상 key값과 value값의 묶음 형태로 저장한다는 것입니다 
# 위의 리스트나 튜플처럼 인덱스 번호로 접근하지 못하고 key를 통해 접근합니다
# print(d[0]) 이 코드를 작성하면
#  print(d[0])
#           ~^^^
# KeyError: 0
# 이런 에러가 나옵니다

d['second'] = 2         # 기존 값 변경
d['new'] = 'new value'  # 새로운 값 추가
print(d) #{'first': 1, 'second': 2, 'inner_dic': {'in_a': 1, 'in_b': 2}, 'new': 'new value'}
# 보다시피 딕셔너리 또한 변경이 자유롭습니다

# 딕셔너리는 우리가 지금까지 써왔던 함수의 인자를 넘길 때도 잘 쓰입니다
# 컴파일에 loss인자에 값을 넘겨주고 할 수 있는 것도 딕셔너리를 통해서 가능합니다
def compile(**kwargs):  #**kwarg 는 key와 value를 몇개든 받아서 딕셔너리형태로 담습니다
    loss_type = kwargs['loss']
    optimizer_type = kwargs['optimizer']
    print(f"{loss_type=}, {optimizer_type=}")
    
compile(loss='mse',optimizer='adam') #loss_type='mse', optimizer_type='adam'
