# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
chunk_iterator=pd.read_csv('D:\machine_usage.csv',names=['m_id','time','cpu_per','mem_per','mem_gps','mpki','ni','no','disk_per'],iterator=True)

mach_chunk1=chunk_iterator.get_chunk(100000000)
'''chunk1.shape
chunk1.head()
chunk1.isnull().sum()
chunk1.columns''''''
''' col_map=dict(zip(['m_1932', '386640', '41', '92', 'Unnamed: 4', 'Unnamed: 5', '43.04',
       '33.08', '5'],['id','time','cpu_per','mem_per','x','y','ni','no','disk_per']))
#chunk1.rename(col_map)  #rename fail
chunk1.drop([chunk1.columns[4],chunk1.columns[5]],axis=1,inplace=True)#delete two cols there are two many null
'''
import datetime
import matplotlib.pyplot as plt
import time

def stamp2time(x):
    s=datetime.datetime.fromtimestamp(x)
    return s.strftime('%Y-%m-%d %H-%M-%S')



#ch_m1932=chunk1[chunk1['m_1932']=='m_1932']
'''''''''''''''''
con_meta=pd.read_csv('D:\container_meta.csv',names=['c_id','m_id','time','app','status','cpu_req','cpu_limit','mem_size'])

'''con_app_1596_df=con_meta[con_meta['app']=='app_1596']
mid_app1596_list=list(con_app_1596_df['m_id'].unique())
total_pd=pd.DataFrame(columns=['mach_time','con_time'])
for i in range(len(app_1552_mid_list)):
    mach_data_i=mach_chunk1[mach_chunk1['m_id']==app_1552_mid_list[i]]
    mach_con_i=con_meta[con_meta['m_id']==app_1552_mid_list[i]]
    null_pd=pd.DataFrame(columns=['mach_time','con_time'])
    data_i_time=list(mach_data_i['time'].values)
    mach_con_i_time=list(mach_con_i['time'].values)
    del mach_data_i,mach_con_i
    max_time=np.max(data_i_time)
    min_time=np.min(data_i_time)
    a_list,i=[],0
    while(i*100+min_time<=max_time):
        a_list.append(i*100+min_time)
    null_pd['mach_time']=pd.Series(a_list)
    null_pd['con_time']=np.zeros((len(a_list),1))
    for j in range(len(a_list)):
        for k in range(len(mach_con_i_time)):
            if((mach_con_i_time[k]>=a_list[j]) and (mach_con_i_time[k]<=a_list[j]+100)):
                null_pd.iloc[j,1]+=1
    total_pd=pd.concat([null_pd,total_pd],axis=0)
  '''


import numpy as np

import pandas as pd

rtm=pd.read_csv(r'C:\Users\Astroknight\Desktop\rtm_df.csv')

tpm=pd.read_csv(r'C:\Users\Astroknight\Desktop\tpm_df.csv')

from sklearn.model_selection import train_test_split
import xgboost as xgb

import lightgbm as lgb

import catboost as cab
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

m1=MinMaxScaler()

m2=MinMaxScaler()

rtm=m1.fit_transform(rtm)

tpm=m2.fit_transform(tpm)
xrtrain,xrtest,yrtrain,yrtest=train_test_split(rtm[:,:-1],rtm[:,-1])

xttrain,xttest,yttrain,yttest=train_test_split(tpm[:,:-1],tpm[:,-1])
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.linear_model import SGDRegressor

from sklearn.neural_network import MLPRegressor
dc=DecisionTreeRegressor()
gs1=GridSearchCV(dc,param_grid={'max_depth':[5,6,7]})
%time gs1.fit(xrtrain,yrtrain) #6.97s
print(mean_squared_error(yrtest,gs1.best_estimator_.predict(xrtest))) # 0.008183702472649936 
gs1.best_params_# {'max_depth': 6}
 ada=AdaBoostRegressor()

gs2=GridSearchCV(ada,param_grid={'n_estimators':[100,200,300],'learning_rate':[0.001,0.01,0.1]})
%time gs2.fit(xrtrain,yrtrain)#27min 5s
print(mean_squared_error(yrtest,gs2.best_estimator_.predict(xrtest)))#0.0007365068107476295
mlp=MLPRegressor()

gs3=GridSearchCV(mlp,param_grid={'max_iter':[200,300,500],'momentum':[0.9,0.95,0.98]})

%time gs3.fit(xrtrain,yrtrain)#15.7 s
print(mean_squared_error(yrtest,gs3.best_estimator_.predict(xrtest)))#0.023764312247502882
lgb_mr=lgb.LGBMRegressor()

gs4=GridSearchCV(lgb_mr,param_grid={'learning_rate':[0.01,0.1],'n_estimators':[200,500]})

%time gs4.fit(xrtrain,yrtrain)# 2min 16s
print(mean_squared_error(yrtest,gs4.best_estimator_.predict(xrtest)))#0.0018185118185325938
xgb_mr=xgb.XGBRegressor()

gs5=GridSearchCV(xgb_mr,param_grid={'learning_rate':[0.01,0.1],'n_estimators':[300,500]})

%time gs5.fit(xrtrain,yrtrain)#12min 56s
print(mean_squared_error(yrtest,gs5.best_estimator_.predict(xrtest)))#0.004098184777428524
cab_mr=cab.CatBoostRegressor()

gs6=GridSearchCV(cab_mr,param_grid={'learning_rate':[0.01,0.1],'n_estimators':[300,500]})

%time gs6.fit(xrtrain,yrtrain)#8min 21s
print(mean_squared_error(yrtest,gs6.best_estimator_.predict(xrtest)))#0.00244213100926382

from keras.models import Input,Model
from keras.layers import Dense,Dropout,Activation,LSTM,RNN,Bidirectional
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping
es=EarlyStopping(verbose=2,monitor='loss',patience=30)
i=Input(shape=[1,5824,])
a=Dense(256,activation='relu')(i)
a=Dropout(rate=0.25)(a)
a=LSTM(units=256,activation='relu',return_sequences=True)(a)
a=Dropout(rate=0.25)(a)
a=LSTM(units=256,activation='relu')(a)
a=Dropout(rate=0.2)(a)
a=LSTM(units=256,activation='relu')(a)
a=Dropout(rate=0.2)(a)
a=Dense(1,activation='relu')(a)
model=Model(inputs=i,outputs=a)
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['accuracy'])
xrtrain=xrtrain.reshape(xrtrain.shape[0],1,xrtrain.shape[1])
history=model.fit(xrtrain,yrtrain,verbose=2,epochs=200,batch_size=10,callbacks=[es])
y_pred=model.predict(xrtest.reshape(xrtest.shape[0],1,xrtest.shape[1]))
print(mean_squared_error(yrtest,y_pred))


del gs1,gs2,gs3,gs4,gs5,gs6
del model
gs1=GridSearchCV(dc,param_grid={'max_depth':[5,6,7]})
%time gs1.fit(xttrain,yttrain) #5.87s
print(mean_squared_error(yttest,gs1.best_estimator_.predict(xttest)))#0.006916988213682063
 ada=AdaBoostRegressor()

gs2=GridSearchCV(ada,param_grid={'n_estimators':[100,200,300],'learning_rate':[0.001,0.01,0.1]})
%time gs2.fit(xttrain,yttrain)#25min 41s
print(mean_squared_error(yrtest,gs2.best_estimator_.predict(xrtest)))#0.011993047809264771

mlp=MLPRegressor()
gs3=GridSearchCV(mlp,param_grid={'max_iter':[200,300,500],'momentum':[0.9,0.95,0.98]})
%time gs3.fit(xrtrain,yrtrain)#13.4 s
print(mean_squared_error(yrtest,gs3.best_estimator_.predict(xrtest)))#0.0672530222340771
lgb_mr=lgb.LGBMRegressor()

gs4=GridSearchCV(lgb_mr,param_grid={'learning_rate':[0.01,0.1],'n_estimators':[200,500]})

%time gs4.fit(xrtrain,yrtrain)# 2min 18s
print(mean_squared_error(yrtest,gs4.best_estimator_.predict(xrtest)))#0.006445514437677034
xgb_mr=xgb.XGBRegressor()

gs5=GridSearchCV(xgb_mr,param_grid={'learning_rate':[0.01,0.1],'n_estimators':[300,500]})

%time gs5.fit(xrtrain,yrtrain)#12min 56s
print(mean_squared_error(yrtest,gs5.best_estimator_.predict(xrtest)))#0.008747605715492551
cab_mr=cab.CatBoostRegressor()

gs6=GridSearchCV(cab_mr,param_grid={'learning_rate':[0.01,0.1],'n_estimators':[300,500]})

%time gs6.fit(xrtrain,yrtrain)#8min 11s
print(mean_squared_error(yrtest,gs5.best_estimator_.predict(xrtest)))#0.0106809299020999202

ii=Input(shape=[1,5824,])
aa=Dense(128,activation='relu')(ii)
aa=Dropout(rate=0.25)(aa)
aa=LSTM(units=128,activation='relu',return_sequences=True)(aa)
aa=Dropout(rate=0.25)(aa)
aa=LSTM(units=128,activation='relu',return_sequences=True)(aa)
aa=Dropout(rate=0.2)(aa)
aa=LSTM(units=128,activation='relu')(aa)
aa=Dropout(rate=0.2)(aa)
aa=Dense(1,activation='relu')(aa)
modell=Model(inputs=ii,outputs=aa)
modell.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['accuracy'])
xttrain=xttrain.reshape(xttrain.shape[0],1,xttrain.shape[1])
history=modell.fit(xttrain,yttrain,verbose=2,epochs=200,batch_size=10,callbacks=[es])
y_pred=modell.predict(xttest.reshape(xttest.shape[0],1,xttest.shape[1]))
print(mean_squared_error(yttest,y_pred))#0.008857739020262781




