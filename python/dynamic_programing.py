# red, green, blue = input("RED GREEN BLUE").split()
# f = open("C:/Study/python/resource/dynamic_programing.txt",'r')
# lines = f.read().splitlines()
# lines = lines.split(sep=' ')
# print(lines)



num = int(input())
data = []
# for line in lines:
#     data.append(line.split(sep=' '))
    
for i in range(num):
    data.append(input().split())

# print(data)
memory = []

# num = int(data[0][0])

for i in range(num):
    red = int(data[i][0])
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