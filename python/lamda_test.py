#map(function, iterator) iterator를 function에 하나씩 넣어서 나온 값을 모아서 다시 list나 tuple로 반환해줌
a = list(range(1,11)) #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mapped_a = list(map(lambda x: x*x, a))
print(f"{a=}\n{mapped_a=}")

#filter(function, iterator) iterator를 function에 넣어 조건의 맞는 요소만이 list나 tuple로 반환
filtered_a = list(filter(lambda x : x%3==0, a))
print(f"{filtered_a=}")

#리스트 표현식
list_expression_a = [[i*i for i in a],[i for i in a if i%3==0]]
print(f"{list_expression_a=}",sep='\n')