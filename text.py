import pandas as pd
sms_raw = pd.read_csv("./input/sms_spam.csv")
sms_raw.describe()
sms_raw.dtypes
sms_raw['type'] = pd.Series(sms_raw['type'].factorize()).iloc[0]
sms_raw['type'].head()
sms_raw.groupby('type').count()
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn import metrics
def to_low(text):
    return text.lower()
def removePunctuation(text):
    return text.translate(None,string.punctuation+('1234567890'))
sms_raw['text']=sms_raw['text'].map(removePunctuation).map(to_low)
sms_raw['text'].head()
count_vect=CountVectorizer(stop_words='english',decode_error='ignore')
sms_text_cved=count_vect.fit_transform(sms_raw['text'])
sms_text_cved.todense().shape
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect=TfidfVectorizer(stop_words='english',decode_error='ignore',min_df=5)
sms_tfidf=tfidf_vect.fit_transform(sms_raw['text'])
sms_tfidf.shape
sms_trainx=sms_tfidf[index_to_select,]
sms_trainy=sms_raw['type'][index_to_select]
sms_testx=sms_tfidf[index_left,]
sms_testy=sms_raw['type'][index_left]
wc=WordCloud()
wc.generate(''.join(sms_raw['text']))
plt.imshow(wc)
plt.show()
plt.axis('off')
from sklearn.naive_bayes import MultinomialNB
mnb_model=MultinomialNB().fit(sms_trainx,sms_trainy)
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
import  seaborn as sns
report=classification_report(sms_testy,mnb_model.predict(sms_testx))
matrix=confusion_matrix(sms_testy,mnb_model.predict(sms_testx))
sns.heatmap(matrix)
score=roc_auc_score(sms_testy,mnb_model.predict(sms_testx))
print(report)
print(score)
