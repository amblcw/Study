'''
클래스를 쓰다보면 여러 비슷한 클래스들을 만들어야 할 때가 있습니다
그렇다면 클래스를 재사용 할수는 없는 걸까요 
이 문제를 해결하기 위해서 상속이란 문법이 존재합니다
상속이란 다른 클래스의 변수와 메서드를 모두 가져오는 기능입니다
이때 받는 쪽을 자식클래스라고하고 주는 쪽을 부모클래스라고 합니다 
'''
class Parent():
    p_a = 'parent'
    bb = 'bb'
    def print_pa(self):
        print(self.p_a)
        
    def print_bb(self):
        print(self.bb)
    
class Child(Parent):
    bb = 'cc'
    def print_bb(self):
        print("this is print_bb")
    
ch = Child()
ch.print_pa()           # parent
ch.print_bb()           # this is print_bb
print(ch.p_a, ch.bb)    # parent cc

'''
이 코드를 보면 Child클래스가 Parent클래스의 p_a와 print_pa를 상속받은 모습을 볼수있습니다
또한 보다시피 부모 클래스가 bb와 print_bb를 갖고있어 이를 상속받았다 하더라도 자식 클래스에서 이걸 재정의 할 수 있습니다
이를 메서드 오버로딩이라고 합니다
'''
class Father():
    def print_aa(self):
        print("father")
        
class Mother():    
    def print_aa(self):
        print("mother")
        
class Son(Father,Mother):
    pass

class Daughter(Mother, Father):
    pass

s = Son()
d = Daughter()

s.print_aa()    # father
d.print_aa()    # mother

'''
보다시피 둘 이상의 클래스를 상속 받을 수도 있습니다
하지만 만일 상속 받을 때 이름이 겹쳐서 충돌이 나는 경우 어떻게 될까요
그 경우 상속을 명시할때 보다 앞에 있는 클래스를 우선합니다
따라서 Son클래스에서는 출력이 "father"가 되었고
Daughter클래스에서는 출력이 "mother"가 된 것입니다

이 외에도 자식 클래스가 부모 클래스의 변수나 메서드를 사용할수도 있습니다
'''
class Shape():
    length = 0
    def print_name(self):
        print(self.__class__.__name__)
    
    def size():
        pass
        
class Circle(Shape):
    radius = 0
    def __init__(self, radius):
        self.length = radius * 2
        self.radius = radius
        
    def print_name(self):
        super().print_name()
        print("type: Circle")
    
    def size(self):
        return self.radius * self.radius * 3.14
    
class Square(Shape):
    def __init__(self, length):
        self.length = length
        
    def print_name(self):
        super().print_name()
        print("type: Square")
    
    def size(self):
        return self.length * self.length
    
c = Circle(3)
s = Square(4)

c.print_name()
# Circle
# type: Circle
s.print_name()
# Square
# type: Square
print(c.size(), s.size())   # 28.26 16
'''
위 코드는 어떻게 부모클래스를 만들고 그 파생형을 만드는지에 대한 예제입니다
위 코드에서는 부모의 print_name 함수를 super().print_name으로 불러와 이를 덧붙이는 식으로 활용하고있습니다
또한 같은 size함수라도 상속받은 자식 클래스의 특성에 맞게 다시 구현하는 모습 또한 볼 수 있습니다
'''