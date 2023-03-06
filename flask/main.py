from flask import Flask, request, render_template
import joblib as jb 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


model = jb.load('nlp/spams.joblib')

app = Flask(__name__)

# load your trained model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    # extract input features from the request data
    spams = pd.read_csv('/Users/da_learner_m1_19/Downloads/spam.csv', encoding="latin-1")
    
    spams['label'] = spams['v1'].map({'ham': 0, 'spam': 1})
    spams['message']=spams['v2']
    spams.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4','v1','v2'], axis=1, inplace=True)
    X = spams['message']
    y = spams['label']
   
    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)  # Fit the Data
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        if my_prediction == 0:
            my_prediction = "✅Not spam✅"
        else:
            my_prediction = "❗️❗️spam❗️❗️"
        
    # return the prediction as a json object
    return render_template('prediction.html',prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True,port=8085)