'''For Predictability of Consumer Behaviour Regarding to Its  Complexity '''
import numpy as np
import pandas as pd
import queue

from scipy.linalg import hankel
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.backend import clear_session

'''Lempel-Ziv complexity'''
def LempelZiv(S):
    n=len(S)
    i=0
    C=u=v=vmax=1
    while (u+v)<n:
        if S[i+v] == S[u+v]:
            v+=1
        else:
            vmax = max(v, vmax)
            i+=1
            v=1
            if i==u:
                C+=1
                u+=vmax
                i=0
                vmax=v
            else:
                v=1
    if v!=1:
        C+=1
    return C

'''F1 and Accuracy for binary classifier'''
def F1metr(x_pred, x_real): #классы: 1 - positive, O - negative
    x_pred, x_real= x_pred.astype(int), x_real.astype(int) 
    tp=len(np.where(x_pred[np.where(x_real==1)]==1)[0])
    tn=len(np.where(x_pred[np.where(x_real==0)]==0)[0])
    fp=len(np.where(x_pred[np.where(x_real==0)]==1)[0])
    fn=len(np.where(x_pred[np.where(x_real==1)]==0)[0])
    if (tp+fp)*(tp+fn)*tp:
        precision, recall = tp/(tp+fp), tp/(tp+fn)
        f1=2*precision*recall/(precision+recall) 
    elif sum(x_pred-x_real):
        f1=0.
    else:
        f1=1.
    if (tp+tn+fp+fn):
        accuracy=(tp+tn)/(tp+tn+fp+fn)*100
    else:
        accuracy=0.
    return f1, accuracy


'''Transactions for a single customer (binary)'''
def GetTrans(data, user, val):
    dfc=data[(data.client==user) & (data.value==val)]
    trans=dfc.groupby('date')['amt'].sum()
    trans=pd.merge(pd.DataFrame({'date':data.date.unique()}), trans, on='date', how='outer').fillna(value=0.).sort_values(by='date')
    trans.reset_index(inplace=True)
    trans.drop('index', axis=1, inplace=True)
    trans['bin']=np.where(trans.amt.values>10., 1, 0)
    return trans

'''integer seconds to string HH:MM:SS'''
def seconds_to_str(seconds):
    mm, ss = divmod(seconds, 60)
    hh, mm = divmod(mm, 60)
    return "%02d:%02d:%02d" % (hh, mm, ss)

'''Huffman encoding'''
class Node:
    def __init__(self, x, k=-1, l=None, r=None, c=''):
        self.freq = x
        self.key = k
        self.left = l
        self.right = r
        self.code = c
    def __lt__(self, otr):
        return self.freq < otr.freq

def huffman_code(data):
    freqTable={}
    nodeList=[]
    que=queue.PriorityQueue()
    codeTable={}
    
    # frequent label init
    for n in data:
        if n in freqTable:
            freqTable[n]+=1
        else:
            freqTable[n]=1
    
    # Huffman tree init
    for k,v in freqTable.items():
        nodeList.append(Node(v,k))
        que.put(nodeList[-1])
        
    # Huffman tree generate
    while que.qsize()>1:
        n1=que.get()
        n2=que.get()
        n1.code='1'
        n2.code='0'
        nn=Node(n1.freq+n2.freq,l=n1,r=n2);
        nodeList.append(nn);
        que.put(nodeList[-1])

    # get Huffman code
    def bl(p,codestr=[]):
        codestr.append(p.code)
        if p.left:
            bl(p.left,codestr.copy())
            bl(p.right,codestr.copy())
        else:
            codeTable[p.key]=''.join(codestr)
    bl(nodeList[-1])
    
    # print Huffman code result
    # print(str(codeTable))
    
    return codeTable

'''Lempel-Ziv data compression'''
def LZW_compress(text, dict_size=8):
    dictionary = {str(i): i for i in range(dict_size)}
    compessed_data = []
    string = ''
    for symbol in text:
        new_string = string + symbol
        if new_string in dictionary:
            string = new_string
        else:
            compessed_data.append(dictionary[string])
            dictionary[new_string] = dict_size
            dict_size += 1
            string = symbol
    if string in dictionary:
        compessed_data.append(dictionary[string])
    return compessed_data, dictionary

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