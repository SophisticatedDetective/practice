impo
rt numpy as np
import pandas as pd
import matplotlib.pyplot as plt
rtm=pd.read_csv('/home/kesci/input/2332324072/rtm_df.csv')
tpm=pd.read_csv('/home/kesci/input/2332324072/tpm_df.csv')
new_rtm=pd.read_csv('/home/kesci/input/2332324072/newrtm_df.csv')
new_tpm=pd.read_csv('/home/kesci/input/2332324072/newtpm_df.csv')
from sklearn.cluster import KMeans

from sklearn.metrics import classification_report, auc, f1_score, roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier
total_df = pd.concat([new_rtm.drop('country', axis=1), new_tpm], axis=1)


km = KMeans(4, max_iter=3000)

km.fit(total_df[total_df.columns[:-1]].values)

total_df.drop('country', axis=1, inplace=True)

td_feat = total_df.values

td_label = pd.Series(km.labels_)

xtrain, xtest, ytrain, ytest = train_test_split(
    td_feat, td_label.values, test_size=0.25)
from keras.models import Model,Input
from keras.layers import Dropout,Dense,LSTM,RNN,Activation
a=Input(shape=(1,11650))
b=Dense(64,activation='relu')(a)
c=LSTM(128,activation='relu')(b)
c=Dropout(0.2)(c)

c=Dense(4,activation='softmax')(c)
m=Model(inputs=a,outputs=c)
xtrain=xtrain.reshape(xtrain.shape[0],1,xtrain.shape[1])
xtest=xtest.reshape(xtest.shape[0],1,xtest.shape[1])
from keras.utils.np_utils import to_categorical
ytrain=to_categorical(ytrain)
ytest=to_categorical(ytest)
m.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
from keras.callbacks import EarlyStopping
history=m.fit(xtrain,ytrain,batch_size=32,verbose=2,epochs=200)
