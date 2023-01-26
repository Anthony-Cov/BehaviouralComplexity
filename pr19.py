import sys
LibPath = '../'
if LibPath in sys.path:
    print('YES')
else:
    print('NO, we\'ll add it now')
    sys.path.append(LibPath)
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

from scipy.linalg import hankel
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.backend import clear_session
from Libraries.Util import F1metr, seconds_to_str, LempelZiv
from time import time, ctime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def GetTrans(user, mcc):
    if type(mcc)==int:
        dfc=data[(data.client==user) & (data.mcc==mcc)]
    else:
        dfc=data[(data.client==user) & (data.value==mcc)]
    trans=dfc.groupby('date')['amt'].sum()
    trans=pd.merge(pd.DataFrame({'date':data.date.unique()}), trans, on='date', how='outer').fillna(value=0.).sort_values(by='date')
    trans.reset_index(inplace=True)
    trans.drop('index', axis=1, inplace=True)
    trans['bin']=np.where(trans.amt.values>10., 1, 0)
    return trans

def MakeSet(ser, lzc, fwd):
    H=hankel(ser)
    X0=H[:-lzc-fwd+1, :lzc]
    X=[]
    for i in range(X0.shape[0]-fwd-1):
        X.append(X0[i:i+fwd+1, :].T)  
    X=np.array(X)
    y=H[:-lzc-2*fwd, lzc+fwd:lzc+2*fwd]
    return X, y

def FitModel(X_train, y_train):
    clear_session()
    xs=X_train.shape
    ys=y_train.shape
    model = Sequential()
    model.add(LSTM(xs[1], input_shape=(xs[1], xs[2]), return_sequences = True)) #level
    model.add(Dropout(0.3))
    model.add(LSTM(xs[1], input_shape=(xs[1], xs[2]), return_sequences = False))
    model.add(Dense(xs[1], activation='tanh'))
    model.add(Dense(ys[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=15, batch_size=1, verbose= 0) 
    return model

datafile='trans19.csv'
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
for i in range(100):
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
    res.to_csv('res_tm.csv', index=False)
    print(i, seconds_to_str(time()-sta),res.iloc[i,1:].values.astype(float).round(2))    
print('Done!', ctime(time()))
