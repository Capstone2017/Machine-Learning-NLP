#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 20:09:00 2016

@author: mac
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics, tree, cross_validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RandomizedLogisticRegression
import glob
import codecs

data = {"train": {"pos": [], "neg": []},
        "test": {"pos": [], "neg": []}}

txt_types = [("train", "neg"), ("train", "pos"), ("test", "neg"), ("test", "pos")]

# with open should add encoding='utf-8', errors='ignore' 
#otherwise failed "'ascii' codec can't decode byte 0xc3 in position 404: ordinal not in range(128)"
for t in txt_types:
    for txt_file in glob.glob("aclImdb/" + t[0] + "/" + t[1] + "/*.txt"):
        with codecs.open(txt_file, "r",encoding='utf-8', errors='ignore') as f:
            text = f.read()
        data[t[0]][t[1]].append(text)
#list(data["train"]["neg"])[0]
# get training + test data

import numpy as np
X_train = data["train"]["pos"] + data["train"]["neg"]
y_train = np.append(np.ones(len(data["train"]["pos"])), np.zeros(len(data["train"]["neg"])))

X_test = data["test"]["pos"] + data["test"]["neg"]
y_test = np.append(np.ones(len(data["test"]["pos"])), np.zeros(len(data["test"]["neg"])))
print(len(X_train), len(y_train))
print(len(X_test), len(y_test))

#tfidf transformation is automately employed in Pipeline
## tfidf = TfidfVectorizer()
## tfidf.fit_transform(X_train)

# build a pipeline - SVC
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                    ('tfidf', TfidfTransformer()),
                    ('clf', OneVsRestClassifier(LinearSVC(random_state=0)))
                     ])

# fit using pipeline
clf = text_clf.fit(X_train, y_train)

# predict
predicted = clf.predict(X_test)
clf.score(X_test, y_test) 

# print metrics
print(metrics.classification_report(y_test, predicted))

#scores = cross_validation.cross_val_score(text_clf, X_train + X_test, np.append(y_train, y_test), cv=5)

#First we have to one-hot encode the text, but let's limit the features to the most common 20,000 words.
from collections import Counter

max_features = 20000
all_words = []

for text in X_train + X_test:
    all_words.extend(text.split())
unique_words_ordered = [x[0] for x in Counter(all_words).most_common()] #x[1] times
word_ids = {}
rev_word_ids = {}
for i, x in enumerate(unique_words_ordered[:max_features-1]):
    word_ids[x] = i + 1  # so we can pad with 0s
    rev_word_ids[i + 1] = x


X_train_one_hot = []
for text in X_train:
    t_ids = [word_ids[x] for x in text.split() if x in word_ids]
    X_train_one_hot.append(t_ids)
   
X_test_one_hot = []
for text in X_test:
    t_ids = [word_ids[x] for x in text.split() if x in word_ids]
    X_test_one_hot.append(t_ids)
    
    
#Now we can use Keras, a popular Theano wrapper, to quickly build an NN classifier.
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU

maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train_one_hot, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test_one_hot, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)