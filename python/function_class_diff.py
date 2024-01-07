#함수 
''' 문법
def 함수이름(인자):
    구현 내용
'''
def counter1(n):
    n -= 1
    return n

print("counter1")

a = 10
while a>0:
    a = counter1(a)
    print(a)
    

#클래스
''' 문법
class 클래스이름:
    맴버변수와 맴버함수들
'''
class Counter:
    __n = 0                 #맴버 변수, __를 붙이면 외부에서 접근불가
    def __init__(self,n):   #생성자
        self.__n = n
        
    def counter2(self):
        self.__n -= 1
        return self.__n
    
    def get_n(self):        #외부에서 접근불가능하기에 접근용 함수
        # print(self.__n)
        return self.__n
    
c = Counter(10)
other_c = Counter(5)

print("counter2")

while c.get_n() > 0 or other_c.get_n() > 0:
    if c.get_n() > 0:
        print(c.counter2() , end=' ')
        
    if other_c.get_n() > 0:
        print(other_c.counter2(), end=' ')
        
    print("")
    
    
'''
함수는 구문들을 하나로 묶어 재사용 가능하게 하는 것입니다 
하지만 위의 구문 과 같이 외부 변수를 필요로 할 때도 있습니다 
이 경우 외부 변수가 실수로 변경당할 수 도 있고 외부 변수도 같이 재사용해야하기에 간편하게 재사용을 한다는 목적과도 멀어집니다

반면에 클래스는 함수에 더해서 변수를 묶어서 재사용을 가능하게 해줍니다
또한 이 변수는 클래스 내부에 존재하기에 외부로 부터 보호받을 수도 있습니다 
더불어서 클래스는 사용할 때 인스턴스를 생성하여 사용하기에 여러개의 counter를 동시에 진행할 수도 있습니다 
만일 함수로 동시에 여러개의 counter를 진행하려면 그만큼 다른 이름의 외부 변수를 만들어 줘야하며 이는 재사용성을 떨어뜨립니다 

따라서 오직 입력값에만 영향을 받는 순수함수 같은 경우엔 함수로 작성하는 것이 유리하지만
외부 변수를 이용해야한다던가 상태를 저장해야하는 경우에는 변수를 같이 묶을 수 있는 클래스가 재사용에 유리하다 생각합니다
'''