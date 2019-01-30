import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train=pd.read_csv('../input/train.csv')#origin by chen du
test=pd.read_csv('../input/test.csv')
sub=pd.read_csv('../input/sample_submission.csv')
target=train['Target']
train.columns
#target.value_counts()如果我们是写成target=train['Target']那么target是Series类型就可以使用value_counts()方法
#target.values
#data=pd.concat([train,test],axis=0)
#data.shape
'''
-----------------处理训练集不同类别样本失衡#train.shape

#len(train_feat)
#len(test_feat)
new_train_feats,new_test_feats=[],[]
for i in train.columns:
    if train[i].isnull().sum()/train.shape[0]>0.6:
        continue
    elif i=='Target':
        continue
    else:
        new_train_feats.append(i)

for i in test.columns:
    if test[i].isnull().sum()/test.shape[0]>0.6:
        continue
    else:
        new_test_feats.append(i)

#len(new_train_feats)
train=train[new_train_feats]
test=test[new_test_feats]
#train.isnull().sum().sort_values()
#test.isnull().sum().sort_values()
#train[['meaneduc','SQBmeaned']].describe()
#train['meaneduc'].dtypes
#train['SQBmeaned'].dtypes

import matplotlib.pyplot as plt
import seaborn as sns
#sns.kdeplot(train['meaneduc'],shade=True)#从频率分布图可以看到，数据分布有很强的集中性，主要集中在6-9之间，我们以7.5作为填充值填充空值
train['meaneduc'].fillna(7.5,inplace=True)
#train['meaneduc'].isnull().sum()
#sns.kdeplot(train['SQBmeaned'],shade=True)#以50为填充值
train['SQBmeaned'].fillna(50,inplace=True)
#train['SQBmeaned'].isnull().sum()
#sns.kdeplot(test['meaneduc'],shade=True)
test['meaneduc'].fillna(8,inplace=True)
#test['meaneduc'].isnull().sum()
#sns.kdeplot(train['SQBmeaned'],shade=True)#以40填充
test['SQBmeaned'].fillna(40,inplace=True)
'''
缺失值处理完毕
'''
'''train.isnull().any().any()
test.isnull().any().any()'''#检验,如果处理正确这两行输出应该是False
#plt.plot(train['meaneduc'].values)问题
'''
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
train_cols=train.columns
test_cols=test.columns
#train_feat.shape
#pd.Series(train_label).value_counts()#现在四个类都是5996个样本
#train_df.head()
#train_df.shape

#data.shape

data_category_cols=['idhogar','dependency','edjefe','edjefa']
train_id=train['Id']
test_id=test['Id']
#train_id
#test_id
train_pure_feat=train[[i for i in train_cols if i!='Id']]
test_pure_feat=test[[i for i in test_cols if i!='Id']]
#train_pure_feat.shape
#test_pure_feat.shape

#data=pd.get_dummies(data)
#data.shape
#data.iloc[0,:]
#data.head()
data=pd.concat([train_pure_feat,test_pure_feat],axis=0)
#data.shape
#data.info()
data=pd.get_dummies(data)
data.shape
train=data.iloc[0:9557,:]
#train.shape
test=data.iloc[9557:,:]
#test.head()
total_train=pd.concat([train,target],axis=1)
total_train.head()

train_x,train_y=ros.fit_sample(train,target)
#train_x.shape
#train_y.shape
from sklearn.model_selection import train_test_split
xtrain,xvalid,ytrain,yvalid=train_test_split(train_x,train_y,test_size=0.2,random_state=0)
#xtrain.shape
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
rdc=RandomForestClassifier(n_estimators=20,n_jobs=-1,random_state=0).fit(xtrain,ytrain)
pred1=rdc.predict(xvalid)
pred1
from sklearn.metrics import mean_squared_error,classification_report,precision_score,recall_score,f1_score,
error=mean_squared_error(pred1,yvalid)
#error
report=classification_report(yvalid,pred1)
#report
result=pd.concat([test_id,pd.Series(pred2)],axis=1)
result
#sub
result.to_csv('result.csv',index=False)
