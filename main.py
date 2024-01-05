import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from string import punctuation

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import os

# Read the data
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

# clean data
train_df.fillna("missing",inplace = True)
test_df.fillna("missing",inplace = True)

# add title and author in text for both Train and Test data
train_df['text'] = train_df['text'] + " " + train_df['title'] + " " + train_df['author']
test_df['text'] = test_df['text'] + " " + test_df['title'] + " " + test_df['author']



# del title, author and id from both
del train_df['title']
del train_df['author']
del train_df['id']

del test_df['title']
del test_df['author']
del test_df['id']

# setup stopwords as set
stop = set(stopwords.words('english'))

# setup puctuation
punctuation = list(punctuation)
stop.update(punctuation)

# perform stemming
stemmer = PorterStemmer()

def stem_text(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            word = stemmer.stem(i.strip())
            final_text.append(word)
    return " ".join(final_text)

train_df['text'] = (train_df['text'].astype(str)).apply(stem_text)
print('Train done!')
test_df['text'] = (test_df['text'].astype(str)).apply(stem_text)
print('Test done!')

# split train data
train_data, val_data, train_labels, val_labels = train_test_split(train_df['text'], train_df['label'], test_size=0.33, random_state=0)

# test df
test_df = test_df['text']

# vectorize
vectorizer = CountVectorizer(min_df=0.0,max_df=1,ngram_range=(1,3))
X = vectorizer.fit_transform(list(train_data))
X_test = vectorizer.transform(list(val_data))
X_test_df = vectorizer.transform(test_df)

# model
model = MultinomialNB()
model.fit(X, train_labels)
print("Model trained")
# predict
y_pred = model.predict(X_test)
# evaluate
print(accuracy_score(y_pred, val_labels))
