#map(function, iterable) iterator를 function에 하나씩 넣어서 나온 값을 모아서 다시 list나 tuple로 반환해줌
a = list(range(1,11)) #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
new_a = list(map(lambda x: x*x, a))
print(f"{a=}\n{new_a=}")