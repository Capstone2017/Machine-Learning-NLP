#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:27:47 2016

@author: mac
"""

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.constraints import maxnorm
from keras.engine.topology import Merge
import pandas as pd

# read data from csv and generate raw train and test data
pinitial_train=pd.read_csv('pgpub_claims_fulltext.csv',delimiter=',',nrows=3000,encoding='utf-8')
pfinal_train=pd.read_csv('patent_claims_fulltext.csv',delimiter=',',nrows=3000,encoding='utf-8')
pinitial_test=pd.read_csv('pgpub_claims_fulltext.csv',delimiter=',',nrows=5000,skiprows=range(1,30000),encoding='utf-8')
pfinal_test=pd.read_csv('patent_claims_fulltext.csv',delimiter=',',nrows=5000,skiprows=range(1,30000),encoding='utf-8')
import numpy as np
X_train = pinitial_train['claim_txt'].tolist()+pfinal_train['claim_txt'].tolist()#.astype(str)
y_train = np.append(np.zeros(len(pinitial_train)),np.ones(len(pfinal_train)))
X_train_add = pinitial_train['claim_no'].tolist()+pfinal_train['claim_no'].tolist()
X_train_add = np.asarray(X_train_add)

#X_test = pinitial_test['pub_no,appl_id,claim_no,claim_txt,dependencies,ind_flg'].tolist()+pfinal_test['pat_no,claim_no,claim_txt,dependencies,ind_flg,appl_id'].tolist()
X_test = pinitial_test['claim_txt'].tolist()+pfinal_test['claim_txt'].tolist()
y_test = np.append(np.zeros(len(pinitial_test)),np.ones(len(pfinal_test)))
X_test_add = pinitial_test['claim_no'].tolist()+pfinal_test['claim_no'].tolist()
X_test_add = np.asarray(X_test_add)

max_features = 20000
maxlen = 200  # cut texts after this number of words (among top max_features most common words)
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
    if type(text) is str:
        all_words.extend(text.split())
unique_words_ordered = [x[0] for x in Counter(all_words).most_common()]
word_ids = {}
rev_word_ids = {}
for i, x in enumerate(unique_words_ordered[:max_features-1]):
    word_ids[x] = i + 1  # so we can pad with 0s
    rev_word_ids[i + 1] = x

X_train_one_hot = []
for k,text in enumerate(X_train):
    if type(text) is str:
        t_ids = [word_ids[x] for x in text.split() if x in word_ids]
        X_train_one_hot.append(t_ids)
    else:
        y_train=np.delete(y_train,k)
        X_train_add=np.delete(X_train_add,k)
        
    
X_test_one_hot = []
for k,text in enumerate(X_test):
    if type(text) is str:
        t_ids = [word_ids[x] for x in text.split() if x in word_ids]
        X_test_one_hot.append(t_ids)
    else:
        y_test=np.delete(y_test,k)
        X_test_add=np.delete(X_test_add,k)

    
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train_one_hot, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test_one_hot, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
emb_model = Sequential()
emb_model.add(Embedding(max_features, 128, input_length=maxlen,dropout=0.2))
emb_model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2)) 


add_model = Sequential()
add_model.add(Dense(4, input_dim=1, init='normal', activation='relu',W_constraint = maxnorm(2)))

model = Sequential()
model.add(Merge([add_model, emb_model], mode='concat', concat_axis=1))


model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit([X_train_add,X_train], y_train, batch_size=batch_size, nb_epoch=1,
          validation_data=([X_test_add,X_test], y_test))
score, acc = model.evaluate([X_test_add,X_test], y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)