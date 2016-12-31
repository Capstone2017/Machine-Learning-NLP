# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 09:59:45 2016

@author: Nero
"""
import pandas as pd
import collections
reader = pd.read_csv('transactions.csv', iterator=True)
loop = True
chunkSize = 100000
chunks = []
while loop:
    try:
        chunkonce = reader.get_chunk(chunkSize)
        chunks.append(chunkonce)
        '''chunks.append(chunk)'''
    except StopIteration:
        loop = False
        print ('1')
chunks=pd.concat(chunks, ignore_index=True)
'''a=pd.DataFrame()
a['filing_date']=chunk['filing_date']
a['patent_issue_date']=chunk['patent_issue_date']'''
'''a=chunks[chunks['event_code'].isin(['CTNF'])]'''
'''b=chunks['event_code']'''
c=chunks['event_code'].value_counts()
d=c.to_frame()
writer = pd.ExcelWriter('output1.xlsx')
d.to_excel(writer,'Sheet1')
writer.save()
