from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics

class CommentClassifier:
    def __init__(self,classifier_type,vector_type):
        self.classifier_type=classifier_type
        self.vector_type=vector_type
    def fit(self,train_x,train_y,max_df):
        text_list=list(train_x)
        if self.vector_type==0:
            self.vector_type=CountVectorizer(max_df,stop_words=stop_words,ngram_range=(1,3)).fit(text_list)
        elif slef.vector_type==1:
            self.vector_type=TfidfVectorizer(max_df,stop_words=stop_words,ngram_range=(1,3),use_idf=False).fit(text_list)
        else:
            self.vector_type=TfidfVectorizer(max_df,stop_words=stop_words,ngram_range=(1,3)).fit(text_list)
        
        self.transformed_train_x=self.vector_type.transform(text_list)
        self.transformed_train_y=self.train_y
        
        if self.classifier_type==1:
            self.model=SVC(kernel='linear',gamma=10**(-5),C=1).fit(self.transformed_train_x,self.transformed_train_y)
        elif self.classifier_type==2:
              self.model=LinearSVC().fit(self.transformed_train_x,self.transformed_train_y)
        else:
              self.model=SGDClassifier().fit(self.transformed_train_x,self.transformed_train_y)
    def predict(self,test_x):
            text_list=list(test_x)
            self.transformed_test_x=self.vector_type.transform(text_list)
            predicted_test=self.model.predict(text_list)
            return predicted_test
    def predict_proba(self,test_x):
        list_text = list(test_x)
        self.array_testx = self.vectorizer.transform(list_text)
        array_score = self.model.predict_proba(self.array_testx)
        return array_score 
        
