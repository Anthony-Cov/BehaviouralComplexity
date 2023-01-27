import os
import numpy as np
import pandas as pd

from nabiim import *

from time import time, ctime

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

datafile='raif_values.csv' #'trans19.csv'
fwd=7

data=pd.read_csv(datafile)
print('Data loaded, off we go!', ctime(time()))
users=list(data.client.value_counts().index)

highlab=['survival','socialization','self_realization']

data
n=len(users)
print(n, 'customers in data')
res=pd.DataFrame(columns=['id']+highlab)
res['id']=['id']*n
sta=time()
split=180
for i in range(n):
    for sssr in highlab: 
        trans=GetTrans(users[i], sssr)
        ser=trans.bin.values
        lzc=LempelZiv(ser)
        if lzc>30: 
            lzc=30
        elif lzc<14: 
            lzc=14
        X, y=MakeSet(ser, lzc, fwd)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test=X[split:], y[split:]
        model=FitModel(X_train, y_train)
        f1=[]
        for j in range (len(y_test)-1):
            z=model.predict(X_test[j:j+1], verbose= 0)
            f1.append(F1metr(z.round(0).astype(int)[0],  y_test[j:j+1][0])[0])
        res.iloc[i, 0]=users[i]
        res.loc[i, sssr]=np.where(np.array(f1)>.75)[0].shape[0]/len(f1)
        del model
        clear_session()
        print('\t',sssr, seconds_to_str(time()-sta), lzc, len(f1) )
    res.to_csv('res_raif.csv', index=False)
    print(i, seconds_to_str(time()-sta),res.iloc[i,1:].values.astype(float).round(2))    
print('Done!', ctime(time()))
