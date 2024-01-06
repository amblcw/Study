# from keras.models import Sequential
# from keras.layers import Dense
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.model_selection import train_test_split
import numpy as np


# path = "C:\\_data\\DACON\\ddarung\\"
# train_csv = pd.read_csv(path+"train.csv",index_col=['id'])  #경로를 적을때 \n같은 경우를 방지하기 위해 \\ 나 /,//도 가능

l = [0.6334175370709949, 0.6762290538295346, 0.6800892792484833, 0.6789109792515216, 0.6456326496207925, 0.6780865517217547, 0.6410305392694677, 0.6763171664691928, 0.6797918753875232, 0.6529530734817269]
print(np.mean(l))

# print(train_csv)
# train_csv = train_csv.fillna(-1)

# print(train_csv.shape)

# nan_index = np.where(np.isnan(train_csv))

# row, colum = nan_index

# for i in range(len(row)):
#     # print(f"({row[i]},{colum[i]})",train_csv.iloc[row[i],colum[i]])
#     pre = train_csv.iloc[row[i]-1,colum[i]]
#     next = train_csv.iloc[row[i]+1,colum[i]]
#     train_csv.iloc[row[i],colum[i]] = (pre + next)/2
#     print(f"({row[i]},{colum[i]})",train_csv.iloc[row[i],colum[i]])
    

#     for row in train_csv.iloc[:,n]:
#         if row == :
#             print("nan")        
        
#         p_pre = pre
#         pre = row
    # print("next")