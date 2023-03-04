# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import classification_report
# import joblib as jb
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.preprocessing import LabelEncoder


# #importing the csv
# spams = pd.read_csv("C:\\Users\\hlala\\Downloads\\spam.csv", encoding="latin-1")
# spams['label'] = spams['v1'].map({'ham': 0, 'spam': 1})
# spams['message']=spams['v2']
# spams.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4','v1','v2'], axis=1, inplace=True)
# #Balancing data
# ham=spams[spams["label"]==0]
# spam=spams[spams["label"]==1]
# ham=ham.sample(spam.shape[0])
# data=spam.append(ham,ignore_index=True)

# tf_vec = TfidfVectorizer()

# #naive = MultinomialNB()

# SVM = SVC(C=1.0, kernel='linear', degree=3 , gamma='auto')

# features = tf_vec.fit_transform(spams['message'])

# X = features
# y = spams['label']

# # Fit the vectorizer to the text data
# vectorizer.fit(text_data)

# # Transform the text data to a bag-of-words representation
# bow_data = vectorizer.transform(text_data)

# # Print the resulting sparse matrix
# print(bow_data)

# cat_data = ["red", "blue", "green", "red", "blue"]

# # Create the label encoder object
# encoder = LabelEncoder()

# # Fit the encoder to the categorical data
# encoder.fit(cat_data)

# # Transform the categorical data to a numeric format
# num_data = encoder.transform(cat_data)

# X=spams["message"]
# Y=spams["label"]
# #fit the model
# cv=CountVectorizer()
# #fit the Data
# X=cv.fit_transform(X)
# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=42)
# #naive bayes classifier
# clf=MultinomialNB()
# clf.fit(X_train,Y_train)
# clf.score(X_test,Y_test)
# y_pred=clf.predict(X_test)
# print(classification_report(Y_test,y_pred))

# filename = 'spams.joblib'
# jb.dump(clf,filename) 
# print('model saved successfully!')
