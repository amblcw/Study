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

class Test_super(metaclass=ABCMeta):
    @abstractmethod
    def test_print(self):
        pass
    
class Test(Test_super):
    def test_print(self):
        print("test success")
        
t = Test()
t.test_print()

    