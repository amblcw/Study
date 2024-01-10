# red, green, blue = input("RED GREEN BLUE").split()
data = []
f = open("C:/Study/python/resource/dynamic_programing.txt",'r')
lines = f.read().splitlines()
# lines = lines.split(sep=' ')
# print(lines)
data = []
for line in lines:
    data.append(line.split(sep=' '))

# print(data)

num = int(data[0][0])

for i in range(1,num+1):
    red = int(data[i][0])
    green = int(data[i][1])
    blue = int(data[i][2])
    print(f"{red=}, {green=}, {blue=}")