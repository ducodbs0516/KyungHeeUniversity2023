import pandas as pd
import os
import numpy as np
ex='e' # a, b, c, d, e (운동 종류)
a = 1 # 1, 5, 10, 15, 20 (전처리 설정)
path ='./original dataset/' # 원본 데이터셋

name = []
dfdf = pd.DataFrame()
# 8가지 주요 관절 각도
coords = ['left shoulder1','right shoulder1', 'left elbow', 'right elbow', 'right shoulder2', 'left shoulder2', 'left hip', 'right hip']

for kk in range(a):
    for ii in range(8):
        name.append(coords[ii] + '_frame%s'%(1+kk))
        
print(name)
for i in range(0,3):
    data_path = path + '%s'%i + '/' + ex
    file_list = os.listdir(data_path)
    for k in range(len(file_list)):
            data_file = pd.read_csv(data_path + '/' + file_list[k])

            df = data_file.iloc[:,:8]
            print(df.isnull().values.any())
            print(data_path + '/' + file_list[k])
            ss = []
            target = []
            for s in range(int(len(df)/a)):
                df_f = df.iloc[s*a:s*a+a]
                df_f = np.array(df_f).ravel()
                ss.append(df_f)
                target.append(i)
            dataset = pd.DataFrame(ss,columns=name)
            targets = pd.DataFrame(target, columns=['target'])
            real = pd.concat([dataset, targets],axis=1)
            dfdf = pd.concat([dfdf, real], axis=0) 
            
dfdf0 = dfdf.dropna(axis=0)
dfdf0.to_csv('C:/Users/KHU/6_10/df/'+ex+'/N_%s'%a+'.csv', index=None)