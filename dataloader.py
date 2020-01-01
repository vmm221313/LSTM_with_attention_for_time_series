import pandas as po
import os
#from tqdm import tqdm_notebook
from sklearn import preprocessing
import numpy as np

def get_data_for_given_ticker(ticker, input_dim, start_date, end_date, train):
    if train:
        df = po.read_csv('merged_data_without_embeddings/' + ticker + 'merged.csv')
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].reset_index(drop = True)
        
        df = df[df['No. Trades'] != 0].reset_index(drop = True)
        
        
        df['Return_class'] = df['class'].shift(1)
        a=df.iloc[0]['Return_class']
        b=df.iloc[0]['Return']
        if type(df.iloc[0]['time']) != 'str':
            a=5
        elif b>=-2*(10**3) and b<-1.11*(10**(-1)):
            a=0
        elif b<-1.39*(10**(-2)):
            a=1
        elif b<1.72*(10**(-2)):
            a=2
        elif b<-1.16*(10**(-1)):
            a=3
        else:
            a=4
            
        df.at[0,'Return_class']=a 
        #print(len(df.columns))
        train = df.drop(['time', 'date', '#RIC', 'class', 'Expected Return'], axis = 1)
        train = train[train.columns[:input_dim]]  
        
        infs = train.index[np.isinf(train).any(1)]
        train = train.drop(infs)
        
        std_scalar_obj = preprocessing.StandardScaler().fit(train)
        train = po.DataFrame(std_scalar_obj.transform(train))
        
        targets = df['class']
        targets = targets.drop(infs)
        
        print(len(train))
        return train, targets
        
    elif train == False:
        df = po.read_csv('merged_data_without_embeddings/' + ticker + 'merged.csv')
        #print(df['date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].reset_index(drop = True)
        df = df[df['No. Trades'] != 0].reset_index(drop = True)
        
        
 
        df['Return_class'] = df['class'].shift(1)
        a=df.iloc[0]['Return_class']
        b=df.iloc[0]['Return']
        if type(df.iloc[0]['time']) != 'str':
            a=5
        elif b>=-2*(10**3) and b<-1.11*(10**(-1)):
            a=0
        elif b<-1.39*(10**(-2)):
            a=1
        elif b<1.72*(10**(-2)):
            a=2
        elif b<-1.16*(10**(-1)):
            a=3
        else:
            a=4
            
        df.at[0,'Return_class']=a 
        
        test = df.drop(['time', 'date', '#RIC', 'class', 'Expected Return'], axis = 1)
        test = test[test.columns[:input_dim]]
        
        infs = test.index[np.isinf(test).any(1)]
        train = test.drop(infs)
        
        std_scalar_obj = preprocessing.StandardScaler().fit(test)
        test = po.DataFrame(std_scalar_obj.transform(test))
        
        targets = df['class']
        targets = targets.drop(infs)
        
        print(len(test))
        return test, targets

# import pandas as po
# df = po.read_csv('merged_data_without_embeddings/' + 'CLc1' + 'merged.csv')
#
# import pandas_profiling
#
# profile = df.profile_report()
# rejected_variables = profile.get_rejected_variables(threshold=0.9)
#
# rejected_variables


