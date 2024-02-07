'''
예외처리
기본은 try except 구조입니다
try:
    예외가 발생할 것 같은 구문
except:
    예외 발생시 처리할 구문 
    
특정 예외에 대해서만 처리할 수도 있으며 예외를 인자로 받아올수있습니다 
try:
    예외가 발생할 것 같은 구문
except 예외이름 as 받을인자이름:
    예외 발생시 처리할 구문
    
예외가 발생하지 않았을 경우도 처리 가능하며 발생했던 안했던 처리도 가능합니다
try:
    예외가 발생할 것 같은 구문
except:
    예외 발생시 처리할 구문
else:
    예외가 발생하지 않았을 때 처리할 구문
finally:
    예외 발생여부와 관계없이 처리할 구문

예외 발생 여부와 관계없이 finally가 호출된다면 왜 사용하는지 궁금할 수 있습니다
finally는 주로 리소스를 반납해야할 때 사용합니다 
만일 파일을 열고 처리하는 과정에서 에러가 났을 때 finally가 없다면
[에러발생 -> 프로그램 종료]로 이어지기에 파일을 닫지 못합니다
except구문으로 처리하면 되지 않냐고 하실 수 있지만 
만일 예외에 따라 다르게 처리해주느라 except구문이 여럿 존재한다면
리소스를 반납하는 구문을 각 except구문마다 적어줘야할것입니다 
하지만 finally구문을 사용하면 이러한 반복을 줄일 수 있습니다

예외를 일부러 발생시킬 수도 있습니다 
예를 들어 반드시 사용자가 처리해줘야 하는 상황에서 에러메세지로 설명과 함께 에러를 던져 처리를 강제하는 경우가 있습니다
'''

import numpy as np

def merge_image(img_iter, fail_stop=False):
    '''
    IDG를 돌려 나눠진 이미지데이터를 병합해주는 함수입니다
    argments:
        img_iter : ImageDataGenerator's iterator
        fail_stop: True면 예외 발생시 함수를 중지시킵니다
    returns:
        data, label
    '''
    x = []          # x값을 담을 리스트
    y = []          # y값을 담을 리스트
    failed_i = []   # 예외가 발생한 번호를 저장할 리스트 
    
    for i in range(len(img_iter)):      # iterator길이만큼 루프합니다 next()호출 과정에서 에러가 발생할 것을 대비하여 간접적인 방식으로 루프합니다
        try:
            xy = img_iter.next()        
            new_x = np.array(xy[0])     
            new_y = np.array(xy[1])
            if i == 0:                  # 바로 병합시키려 하면 shape가 동일하지않다는 오류가 나기에 최초 1회는 그대로 대입
                x = new_x
                y = new_y
                continue
            
            if len(new_y.shape) == 1:   # 만약 new_y.shape = (N,) 형태라면, 즉 이진분류라면
                x = np.vstack([x,new_x])
                y = np.hstack([y,new_y])
            else:                       # 이진분류가 아니니 다중분류라면
                x = np.vstack([x,new_x])
                y = np.vstack([y,new_y])
                
        except Exception as e:
            print("faied i: ",i)
            failed_i.append(i)
            if fail_stop:         # fail_stop이 True로 설정되어있으면 발생한 예외를 그대로 던진다
                raise e
                
        print("i: ",i)
        print(f"{x.shape=}\n{y.shape=}")    
        
    print("failed i list: ",failed_i)
    return x, y

from keras.preprocessing.image import ImageDataGenerator
'''
image_size는 꼭 150,150,3으로 즉 컬러로 해야한다
'''

BATCH_SIZE = int(500)
IMAGE_SIZE = int(150)

path = "C:\\_data\\image\\rps\\"

xy_data_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest'
)
xy_data = xy_data_gen.flow_from_directory(
    path,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
)

x, y = merge_image(xy_data)

save_path = path+f"data_{IMAGE_SIZE}px"
np.save(save_path+"_x.npy",arr=x)
np.save(save_path+"_y.npy",arr=y)

# 출력
# Found 2520 images belonging to 3 classes.
# i:  1
# x.shape=(1000, 150, 150, 3)
# y.shape=(1000, 3)
# i:  2
# x.shape=(1500, 150, 150, 3)
# y.shape=(1500, 3)
# i:  3
# x.shape=(2000, 150, 150, 3)
# y.shape=(2000, 3)
# i:  4
# x.shape=(2500, 150, 150, 3)
# y.shape=(2500, 3)
# i:  5
# x.shape=(2520, 150, 150, 3)
# y.shape=(2520, 3)
# failed i list:  []