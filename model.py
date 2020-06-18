import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import HashingVectorizer
import itertools
import numpy as np 

#%%HTML
#<textarea name="myTextBox" cols="50" rows="5">
#ENter gtext
#</textarea>
#<input type="submit" value="Submit">


data=pd.read_csv('C:/Users/HP/Downloads/news.csv')

data.head()

data.shape

import string
string.punctuation

def removePun(text):
  text_no="".join([char for char in text if char not in string.punctuation])
  return text_no

data['body_text_clean'] = data['text'].apply(lambda x: removePun(x))


import re

def token(text):
  tokens=re.split('\W+',text)
  return tokens

data['body_text_token'] = data['body_text_clean'].apply(lambda x: token(x.lower()))

import nltk

ps = nltk.PorterStemmer()

def stemming(tokenizedText):
  text = [ps.stem(word) for word in tokenizedText]
  return text

data['body_text_stemmed'] = data['body_text_token'].apply(lambda x: stemming(x))

data.head()

y=data.label

data.drop("label",axis=1)

X_train,X_test,y_train,y_test = train_test_split(data['text'],y,test_size=0.3,random_state=50)

def confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

countVe=CountVectorizer(stop_words='english')
countTrain=countVe.fit_transform(X_train)
countTest = countVe.transform(X_test)
cls=CountVectorizer()
cls.fit(X_train,y_train)

tfidfVe = TfidfVectorizer(stop_words='english',max_df=0.5)
tfidfTrain =tfidfVe.fit_transform(X_train)
tfidfTest = tfidfVe.transform(X_test)

count_df = pd.DataFrame(countTrain.A, columns=countVe.get_feature_names())

tfidf_df = pd.DataFrame(tfidfTrain.A, columns=tfidfVe.get_feature_names())

difference = set(count_df.columns) - set(tfidf_df.columns)

print(difference)

print(count_df.equals(tfidf_df))

print(count_df.head())

print(tfidf_df.head())

clf = MultinomialNB() 

clf.fit(tfidfTrain, y_train)                      
pred = clf.predict(tfidfTest)
y_pred=clf.predict(tfidfTrain)
train_acc=metrics.accuracy_score(y_train,y_pred)
print("Training accuracy:    %.03f"% train_acc)                     
score = metrics.accuracy_score(y_test, pred)
print("Testing accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
#confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)

clf = MultinomialNB() 

clf.fit(tfidfTrain, y_train)                      
pred = clf.predict(tfidfTest)
y_pred=clf.predict(tfidfTrain)
train_acc=metrics.accuracy_score(y_train,y_pred)
print("Training accuracy:    %.03f"% train_acc)                     
score = metrics.accuracy_score(y_test, pred)
print("Testing accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
#confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)

linear_clf = PassiveAggressiveClassifier(max_iter=50)

linear_clf.fit(tfidfTrain, y_train)
y_pred=linear_clf.predict(tfidfTrain)
train_acc=metrics.accuracy_score(y_train,y_pred)
print("Training accuracy:    %.03f"% train_acc)
pred = linear_clf.predict(tfidfTest)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=["FAKE", "REAL"])
#confusion_matrix(cm, classes=["FAKE", "REAL"])
print(cm)

clf = MultinomialNB(alpha=0.2)

last_score = 0
for alpha in np.arange(0,2,.2):
    nb_classifier = MultinomialNB(alpha=alpha)
    nb_classifier.fit(tfidfTrain, y_train)
    pred = nb_classifier.predict(tfidfTest)
    score = metrics.accuracy_score(y_test, pred)
    if score > last_score:
        clf = nb_classifier
    print("Alpha: {:.2f} Score: {:.5f}".format(alpha, score))


def most_informative_feature_for_binary_classification(vectorizer, classifier, n=100):       
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()                                            
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)

    print()

    for coef, feat in reversed(topn_class2):
        print(class_labels[1], coef, feat)


most_informative_feature_for_binary_classification(tfidfVe, linear_clf, n=20)
feature_names = tfidfVe.get_feature_names()

sorted(zip(clf.coef_[0], feature_names), reverse=True)[:10]

sorted(zip(clf.coef_[0], feature_names))[:10]   

tokens_with_weights = sorted(list(zip(feature_names, clf.coef_[0])))

hashVe = HashingVectorizer(stop_words='english', alternate_sign=False)
hashTrain = hashVe.fit_transform(X_train)
hashTest = hashVe.transform(X_test)

clf = MultinomialNB(alpha=.01)

clf.fit(hashTrain, y_train)
pred = clf.predict(hashTest)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
#plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)

clf = PassiveAggressiveClassifier(max_iter=50)    

clf.fit(hashTrain, y_train)
pred = clf.predict(hashTest)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
#confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)

