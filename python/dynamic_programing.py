
num = int(input())  #반복문 횟수 결정용
data = []
    
for i in range(num):
    data.append(input().split())    #공백을 기준으로 분리해서 리스트로 넣어줌, type은 str

# print(data)
memory = []

'''
RED를 선택하고 싶으면 이전 선택은 GREEN이나 BLUE가 되어야한다
만일 GREEN을 선택하기 까지 최선의 선택의 결과와 BLUE를 선택하기까지 최선의 결과중 더 작은쪽을 고른다면
그렇게 고르고 RED를 고른건 최선의 선택이란 가정으로 작성하였다 

따라서 memory는 각 행당 3개의 열(red, green, blue)를 가지며 해당 위치에는 그 떄까지의 최선의 값이 들어간다

1행에는 이전 선택이 존재하지 않으므로 첫번쨰 rgb값이 들어간다
2행에는 본인과 다른 1행의 색중 더 작은 값과 자신의 값을 더한 것이 들어간다
n행에는 본인과 다른 n-1행의 색중 더 작은 값과 자신의 값을 더한 것이 들어간다 
'''

for i in range(num):
    red = int(data[i][0])   #str형태이므로 int로 변경 필요
    green = int(data[i][1])
    blue = int(data[i][2])
    # print(f"{red=}, {green=}, {blue=}")
    if i==0:
        memory.append([red,green,blue])
    else:
        memorid_red = int(memory[i-1][0])
        memoried_green = int(memory[i-1][1])
        memoried_blue = int(memory[i-1][2])
        
        # print(f"{memorid_red=},{memoried_green=},{memoried_blue=}")
        
        new_red = min(memoried_green,memoried_blue)+red
        new_green = min(memorid_red,memoried_blue)+green
        new_blue = min(memorid_red,memoried_green)+blue
        
        memory.append([new_red,new_green,new_blue])
        
    # print(memory)

print(min(memory[-1]))