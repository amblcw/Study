def make_closer(init_a):
    a = init_a
    
    return lambda x : x+a

plus_10 = make_closer(10)

print(plus_10(3))

def make_count_down(start_count):
    count = start_count
    def reducer():
        nonlocal count
        count -= 1
        return count+1
        
    return reducer

count_down = make_count_down(10)

for i in range(10):
    print(count_down())