# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv(r'C:\Users\DELL\Downloads\2ML\Machine Learning A-Z\Part 7 - Natural Language Processing\Section 36 - Natural Language Processing\Natural_Language_Processing\Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

dataset.groupby('Liked').describe()

dataset['Length'] = dataset['Review'].apply(len)

dataset.Length.describe()

dataset[dataset['Length'] == 149]['Review'].iloc[0]

dataset['Length'].plot(bins=70, kind='hist') 

dataset.hist(column='Length', by='Liked', bins=40,figsize=(12,4))

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm