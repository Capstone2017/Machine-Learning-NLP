#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 00:51:09 2016

@author: Yu Zhipeng
"""

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding,Flatten,Reshape
from keras.layers import LSTM, SimpleRNN, GRU
from keras.constraints import maxnorm
import pandas as pd

# read data from csv and generate raw train and test data
pinitial_train=pd.read_csv('pgpub_claims_fulltext.csv',delimiter='\t',nrows=3000,encoding='utf-8')
pfinal_train=pd.read_csv('patent_claims_fulltext.csv',delimiter='\t',nrows=3000,encoding='utf-8')
pinitial_test=pd.read_csv('pgpub_claims_fulltext.csv',delimiter='\t',nrows=3000,skiprows=range(1,3000),encoding='utf-8')
pfinal_test=pd.read_csv('patent_claims_fulltext.csv',delimiter='\t',nrows=3000,skiprows=range(1,3000),encoding='utf-8')
import numpy as np
X_train = pinitial_train['pub_no,appl_id,claim_no,claim_txt,dependencies,ind_flg'].tolist()+pfinal_train['pat_no,claim_no,claim_txt,dependencies,ind_flg,appl_id'].tolist()#.astype(str)
y_train = np.append(np.zeros(len(pinitial_train)),np.ones(len(pfinal_train)))

X_test = pinitial_test['pub_no,appl_id,claim_no,claim_txt,dependencies,ind_flg'].tolist()+pfinal_test['pat_no,claim_no,claim_txt,dependencies,ind_flg,appl_id'].tolist()
y_test = np.append(np.zeros(len(pinitial_test)),np.ones(len(pfinal_test)))

maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 64
'''
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train_one_hot, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test_one_hot, maxlen=maxlen)
'''
from collections import Counter

max_features = 20000
all_words = []

for text in X_train + X_test:
    all_words.extend(text.split())
unique_words_ordered = [x[0] for x in Counter(all_words).most_common()]
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

    
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train_one_hot, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test_one_hot, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
embedding_size=128
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen,dropout=0.2))
print('first layer...')
#model.add(Reshape((1, maxlen, embedding_size)))
model.add(Flatten())
'''
model.add(Dense(1000, input_shape=(embedding_size*max_features,)))
print('seconde layer...')
model.add(Dropout(0.2))
'''
model.add(Dense(500, init='normal', activation='relu',W_constraint = maxnorm(2)))
model.add(Dropout(0.2))
model.add(Dense(200, init='normal', activation='relu',W_constraint = maxnorm(2)))
model.add(Dropout(0.2))
model.add(Dense(200, init='normal', activation='relu',W_constraint = maxnorm(2)))
model.add(Dropout(0.2))
model.add(Dense(100, init='normal', activation='relu',W_constraint = maxnorm(2)))
model.add(Dropout(0.2))
model.add(Dense(1, init='normal', activation='sigmoid',W_constraint = maxnorm(2)))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=5,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
