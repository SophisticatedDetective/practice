import warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("./input"))
import pandas as pd 
#读入数据
train = pd.read_csv('./input/labeledTrainData.csv',delimiter = '\t')
test = pd.read_csv('./input/testData.csv',delimiter = '\t')
train.shape, test.shape
train.head()
train['review'][0]
print([i for i in train['review'][0].split() if not i in string.punctuation if not i in ['/><br',]])
print ("number of rows for sentiment 1: {}".format((train[train.sentiment == 1]).shape[0]))
print ( "number of rows for sentiment 0: {}".format((train[train.sentiment == 0]).shape[0]))
train.groupby('sentiment').describe()
train['length'] = train['review'].apply(len)
import matplotlib.pyplot as plt
%matplotlib inline
plt.hist(train['length'],bins=30)
train[train['length'] == 100]['review'].iloc[0]
train.hist(column='length', by='sentiment', bins=100,figsize=(8,4))
import nltk 
nltk.download("stopwords")
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
r=BeautifulSoup(train['review'][0]).get_text()
print(r)
def preprocess_text(text):
    text=BeautifulSoup(text).get_text()
    alpha_words=re.sub('[^a-zA-Z]','  ',text)
    text=text.lower().split()
    new_text=[w for w in text if not w in stopwords.words('english')]
    return new_text
#text='ww ! </>'  #An example
#preprocess_text(text)
train['clean_review']=train['review'].apply(preprocess_text)
train['length_clean_review']=train['clean_review'].apply(len)
print(train[train['length_clean_review']==100]['review'].iloc[0])
print('\n ----After cleaned----\n')
print(train[train['length_clean_review']==100]['clean_review'].iloc[0])
from wordcloud import WordCloud
wc_cleaned= WordCloud(width = 500, height = 500,  background_color = 'black').generate(
                        ''.join(str(train['clean_review'])))

plt.figure(figsize = (8,8))
plt.imshow(wc_cleaned)
plt.axis('off')
plt.show()
wc_noclean=WordCloud(width=500,height=500,backgroud_color='white').generate(''.join(str(train['review'])))
plt.figure(figsize=(8,8))
plt.imshow(wc_noclean)
plt.axis('off')
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train['review'], train['sentiment'], test_size=0.4, random_state=101)
from sklearn.feature_extraction.text import CountVectorizer
cv_text_model = CountVectorizer(analyzer=clean_text,binary = True,max_features=5000).fit(train['review']) #如果已经用自定义文本处理函数做过处理就不需要指定analyzer参数
print(cv_text_model.vocabulary_)
cv_transed_review1 =cv_text_model.transform(train['review'].iloc[0])
print(cv_transed_review1.todense())
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(norm="l2",smooth_idf=True).fit(cv_transed_review1)
tfidf1 = tfidf_transformer.transform(cv_transed_review1)
from sklearn.metrics import classification_report
def pred(predicted,compare):
    cm = pd.crosstab(compare,predicted)
    TN = cm.iloc[0,0]
    FN = cm.iloc[1,0]
    TP = cm.iloc[1,1]
    FP = cm.iloc[0,1]
    print(classification_report(compare,predicted))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
params={'n_estimators':[50,100,150],'learning_rate':[0.001,0.01,0.1]}
rfc = RandomForestClassifier(random_state=42)
pipeline = Pipeline([('cv_text', CountVectorizer(analyzer=clean_text,binary=True,max_features=5000)),  
    ("tfidf" , TfidfTransformer(norm="l2",smooth_idf=True)),
    ("classifier",GridSearchCV(rfc,params, cv=5, scoring='r2', n_jobs=4, verbose=1))])
pipeline.fit(X_train,y_train)
predictions = pipeline_bool.predict(X_train)
pred(predictions,y_train)
predictions = pipeline.predict(X_test)
pred(predictions,y_test)
test['sentiment'] = pipeline.predict(test['review'])
output = test[['id','sentiment']]
print(output)
#with open('...','w') as f_to_write:
#f_to_write.write(output)
