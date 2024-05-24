'''
오버로딩이란 부모클래스로부터 상속받은 메서드를 재정의 하는 것을 말한다
'''

class Parnet():
    def func():
        print("Parent func")
        
class Chinld(Parnet):
    def func(a:int):
        print("input: ",a)
        
'''
반면에 오버라이딩은 같은 이름의 함수지만 다른 인자를 받는 동명의 함수를 정의 하는 것을 말한다
파이썬에선 허용하지 않는다
'''

def func():
    print("None")
    
def func(a:str):
    print("input: ",a)
    
func('sss')
func()