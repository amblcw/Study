from abc import *

# def test(x,y,a,b):
#     new_y = a * x + b
#     loss = new_y - y
#     return loss

# def mediate():
#     pass

# def return_test(a,b):
#     return a+b, a-b

# q, w = return_test(1,2)
# print(q,w)

class Test_super(metaclass=ABCMeta):    #추상클래스, abc 임포트 해야 사용가능
    @abstractmethod                     #추상메소드, 상속받은 클래스는 구현하지 않을 시 사용 불가능
    def test_print(self):
        pass
    
class Test(Test_super):
    def test_print(self):
        print("test success")
        
    def test_arguments(self,*args, **kwargs):   #*args: 여러인자받기, **kwargs: 딕셔너리 형태로 키워드와 같이 받기 (꼭 이름이 args, kwargs일 필요 없음)
        if args != ():
            print("args:")
            for i in args:
                print(f"{i}", end=' ')
            print('\n')
        
        if kwargs != ():
            print("kwargs:")
            for name, value in kwargs.items():
                print(f"{name}={value}", end=' ')
            print('\n')
                
                
t = Test()
t.test_print()
t.test_arguments(1,2,3,4,a='apple',b='bus')

    