# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 22:13:13 2016

@author: Nero
"""

import collections
import pandas as pd
import csv

counters1 = collections.Counter()
counters2 = collections.Counter()

for line in open('transactions.csv'):
    parts = line.split(',')
    '''counters1[parts[0]] += 1'''
    if(parts[1]=='CTNF'):
       if(counters2.__contains__(parts[0])):
           continue
       counters2[parts[0]]=parts[2]
       
'''Export Allevent_code and Allapplication_ID in excel'''
'''df1 = pd.DataFrame.from_dict(counters1, orient='index').reset_index()
writer1 = pd.ExcelWriter('Allevent_code.xlsx')
df1.to_excel(writer1,'Sheet1')
writer1.save()
df2 = pd.DataFrame.from_dict(counters2, orient='index').reset_index()
writer2 = pd.ExcelWriter('Allapplication_ID.xlsx')
df2.to_excel(writer2,'Sheet1')
writer2.save()'''

'''Export csv through dataframe'''
'''df1 = pd.DataFrame.from_dict(counters1, orient='index').reset_index()
df2 = pd.DataFrame.from_dict(counters2, orient='index').reset_index()
df1.to_csv('id')
header=['index','application','recorded_date']
df2.to_csv('CTNF_date',columns = header)'''

'''Export csv directly'''
with open('CTNF_date.csv', 'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Index','Application_ID','Recorded_date_CTNF'])
    i=0;
    for key, count in counters2.items():
        i+=1
        writer.writerow([i,key,count])
    f.close()
