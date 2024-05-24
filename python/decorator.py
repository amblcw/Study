def deco(func):
    def wrapper(*args,**kwargs):
        print(f'{func.__name__} is started')
        func(*args,**kwargs)
        print(f'{func.__name__} is ended')
    return wrapper

def just_print_AAA():
    print("AAA")    
    
deco1 = deco(just_print_AAA)
deco1()
# just_print_AAA is started
# AAA
# just_print_AAA is ended

@deco
def just_print_BBB():
    print("BBB")    
    
just_print_BBB()
# just_print_BBB is started
# BBB
# just_print_BBB is ended

'''
데코레이터는 이름 그대로 다른 함수를 인자로 받아서 인자로 받은 함수를 꾸며주어 기능을 추가하는 등의 기능을 하는 함수를 일컫습니다
기본적으로는 just_print_AAA함수의 경우 같이 직접 함수를 인자로 넘겨주어 사용해야하지만
파이썬에서는 @를 통해 데코레이션 문법을 지원해주기 때문에 just_print_BBB처럼 @deco를 통해 간단하게 적용해줄 수 있습니다
'''