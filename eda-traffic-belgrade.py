#!/usr/bin/env python
# coding: utf-8

# In[103]:


# %load ../../misc/utils/import.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import datetime

from pandas_ods_reader import read_ods

#Display Settings
pw = 16
ph = 9
matplotlib.rcParams['figure.figsize'] = (pw, ph)

#Pandas Dsiplay
pd.options.display.max_rows = 100
pd.options.display.max_columns = 200
pd.options.display.max_colwidth = 200

# import warnings
# warnings.filterwarnings('ignore')


# In[150]:


#Read df
file_name = 'NEZ_OPENDATA_2018_20190125.ods' # year 2018

columns = ['id', 'date', 'long', 'lat', 'acc_outcome' ,'acc_type', 'description']
df = read_ods(file_name, 1, columns=columns)
print("Number of accidents {}".format(len(df)))


# In[151]:


#Sample
df.sample(5, random_state=23)


# In[152]:


#To date-time
df['date'] = pd.to_datetime(df['date'])

df = df.sort_values('date')


# In[153]:


#Check Duplicates
dupl_ids = df[df.duplicated(subset=['id'])]['id']

df.set_index('id').sort_index().loc[dupl_ids].head(4)


# In[154]:


#Drop Duplictates
print("Before duplicates removal {}".format(len(df)))
df = df.drop_duplicates(subset=['id'])
print("After duplicates removal {}".format(len(df)))


# ## Accidents Outcomes

# In[160]:


#Plot
order = df['acc_outcome'].value_counts().index

ax = sns.countplot(df['acc_outcome'], order=order, color='#f57542');

ax.set_title('Accident Outcomes Distribution')
plt.xticks(rotation=45);


# In[161]:


df['acc_outcome'].value_counts()


# In[162]:


df[df['acc_outcome'] == 'Sa poginulim']


# ## Accident Types

# In[163]:


#Plot
order = df['acc_type'].value_counts().index

ax = sns.countplot(df['acc_type'], order=order, color='#32a89b');

ax.set_title('Accident Type Distribution')
plt.xticks(rotation=45);


# In[164]:


#Plot
topl = df[df['acc_outcome'] == 'Sa poginulim']
order = df['acc_type'].value_counts().index

ax = sns.countplot(topl['acc_type'], order=order, color='#eb4c34');

ax.set_title('Accident Type Distribution - Accidents with Casualties')
plt.xticks(rotation=45);


# ## Seasonality of Accidents

# In[165]:


#Seasonal df
ses_df = df.set_index('date')
ses_df['count'] = 1

#Resample
ses_df = ses_df.resample('1m')[['count']].sum()


# In[175]:


#Plot
ax = ses_df.plot();

ax.set_title('Number of Accidents Throughout The Year');


# In[176]:


#Weekday
day_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['day_of_week'] = df['date'].dt.dayofweek.map(day_map)


# In[177]:


#Plot
ax = sns.countplot(df['day_of_week'], color='#46d4d1');

ax.set_title('Day Of Week');


# ## GeoLoc 

# In[ ]:




