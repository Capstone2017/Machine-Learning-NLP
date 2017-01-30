#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 00:21:37 2017

@author: mac
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
i=0
y_train=[]
for line in open('/Volumes/Zhipeng/patent_dataset/paired_newclaims_2012to2013.txt'):
    if i%2==0:
        y_train.append(line)
    i+=1
'''   
i=0
x_train=[]
for line in open('/Volumes/Zhipeng/patent_dataset/paired_oldclaims_2012to2013.txt'):
    if i%2==0:
        x_train.append(line)
    i+=1
'''
vectorizer = TfidfVectorizer(min_df=1)
X=vectorizer.fit_transform(y_train)
#pca = decomposition.PCA(n_components=1)
#pca.fit(X)
#data_reduced_to_one_dimension = pca.transform(X)
#sim=(X * X.T).A


#subX=X[0:100,:]
#sub_sim=(subX * subX.T).A

sim0=(X[0,:]*X.T).A
top10=np.argsort(sim0)[0][-10:]