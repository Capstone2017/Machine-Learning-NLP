# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 19:16:40 2016

@author: Nero
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
reader = pd.read_csv('application_data.csv', iterator=True)
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
chunk=pd.concat(chunks, ignore_index=True)
a=pd.DataFrame()
a['filing_date']=chunk['filing_date']
a['patent_issue_date']=chunk['patent_issue_date']
a=a[pd.notnull(a['filing_date'])] 
a=a[pd.notnull(a['patent_issue_date'])]
timebegins=pd.to_datetime(a['filing_date'], format="%Y-%m-%d")
timeends=pd.to_datetime(a['patent_issue_date'], format="%Y-%m-%d")
timespanyears=timeends-timebegins
timespanyears=timespanyears.to_frame()
timespanyears[0]=timespanyears[0].astype(np.int64)
timespanyears=timespanyears[timespanyears[0]>0]
timespanyears[0]=timespanyears[0]/(1000000000*3600*24*365)
''' Remove for loop by remove nan from a and convert it to datetime 
timespanyears=[]
for index in range(a.shape[0]):
    if(type(a['filing_date'][index])==type('string') 
    and type(a['patent_issue_date'][index])==type('string')):
        filing_dt=datetime.datetime.strptime(a['filing_date'][index],'%Y-%m-%d')
        issue_dt=datetime.datetime.strptime(a['patent_issue_date'][index],'%Y-%m-%d')
        timespan=issue_dt-filing_dt
        x=timespan.total_seconds()/(3600*24*365)      
        if(x>=0):
           timespanyears.append(x)
'''
application_number=[]
invention_subject_matter=[]
disposal_type=[]
'''z=np.histogram(timespanyears,bins=[0,1,2,3,4,5,6,7,8,9,10])'''
plt.hist(timespanyears[0],bins=np.linspace(0,8))
plt.title("Application time span(5,619,688/9,817,693)")
plt.xlabel("Years")
plt.ylabel("Amount")
plt.savefig('spandistribution',dpi=1000)
plt.show()
