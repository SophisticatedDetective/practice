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
 #you can try to modify the raw data and then put the data into the following process part from line 43 to 54       
train_x, test_x, train_y, test_y = train_test_split(data_cutted['Comment'], data_cutted['Class'],test_size=0.3,random_state=0)                                                        test_size=0.2, random_state=4)
classifier_list = [1,2,3]
vector_list = [0,1,2]
for classifier_type in classifier_list:
    for vector_type in vector_list: 
           comment=CommentClassifier(classifier_type,vector_type)
            comment.fit(train_x,train_y,0.8)
            if classifier_type==0:
                predicted=comment.predict(text_x)
                predicted_probability=comment.predict_proba(test_x)
                print(metrics.classification_report(test_y,predicted))
                print(metrics.confusion_matrix(test_y,predicted))
train_x = data_bi['Comment']
train_y = data_bi['Class']
def labelizeReviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '{0:s}_{1:f}'.foramt(label_type, i)
        labelized.append(TaggedDocument(v.split(" "), [label]))
    return labelized
train_x = labelizeReviews(train_x, "TRAIN")
size = 300
all_data = []
all_data.extend(train_x)
model = Doc2Vec(min_count=1, window=8, size=size, sample=1e-4, negative=5, hs=0, iter=5, workers=8)
model.build_vocab(all_data)
for epoch in range(10):
    model.train(train_x)
pos,neg = [],[]
for i in range(0,len(train_x)):
    pos.append(model.docvecs.similarity("TRAIN_0","TRAIN_{}".format(i)))
    neg.append(model.docvecs.similarity("TRAIN_1","TRAIN_{}".format(i)))
data_bi['PosSim'] = pos
data_bi['NegSim'] = neg
