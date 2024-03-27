
target = [-30,-20,-10,1,10,20,30,50]

for w in target:
    s1 = (w*1-1)*(w*1-1)
    s2 = (w*2-2)*(w*2-2)
    print(f"{s1} | {s2} | mse={(s1+s2)/2}")

