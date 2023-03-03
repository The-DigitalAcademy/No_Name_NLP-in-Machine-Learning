import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

#importing the csv
spams= pd.read_csv("")
spams["label"]=spams["v1"].map({'ham':0,'spam':1})
spams["Message"]=spams["v2"]
X=spams["Message"]
Y=spams["label"]
#fit the model
cv=CountVectorizer()
#fit the Data
X=cv.fit_transform(X)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=42)
#naive bayes classifier
clf=MultinomialNB()
clf.fit(X_train,Y_train)
clf.score(X_test,Y_test)
y_pred=clf.predict(X_test)
print(classification_report(Y_test,y_pred))
