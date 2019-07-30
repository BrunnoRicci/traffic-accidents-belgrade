#!/usr/bin/env python
# coding: utf-8

# In[8]:


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


# In[9]:


#Read df
file_name = 'NEZ_OPENDATA_2018_20190125.ods' # year 2018

columns = ['id', 'date', 'long', 'lat', 'acc_outcome' ,'acc_type', 'description']
df = read_ods(file_name, 1, columns=columns)
print("Number of accidents {}".format(len(df)))


# In[10]:


#Sample
df.sample(5, random_state=23)


# In[11]:


#To date-time
df['date'] = pd.to_datetime(df['date'])

df = df.sort_values('date')


# In[12]:


#Check Duplicates
dupl_ids = df[df.duplicated(subset=['id'])]['id']

df.set_index('id').sort_index().loc[dupl_ids].head(4)


# In[13]:


#Drop Duplictates
print("Before duplicates removal {}".format(len(df)))
df = df.drop_duplicates(subset=['id'])
print("After duplicates removal {}".format(len(df)))


# ## Accidents Outcomes

# In[14]:


#Plot
order = df['acc_outcome'].value_counts().index

ax = sns.countplot(df['acc_outcome'], order=order, color='#f57542');

ax.set_title('Accident Outcomes Distribution')
plt.xticks(rotation=45);


# In[15]:


df['acc_outcome'].value_counts()


# In[92]:


df[df['acc_outcome'] == 'Sa poginulim'].sample(5, random_state=23)


# ## Accident Types

# In[17]:


#Plot
order = df['acc_type'].value_counts().index

ax = sns.countplot(df['acc_type'], order=order, color='#32a89b');

ax.set_title('Accident Type Distribution')
plt.xticks(rotation=45);


# In[18]:


#Plot
topl = df[df['acc_outcome'] == 'Sa poginulim']
order = df['acc_type'].value_counts().index

ax = sns.countplot(topl['acc_type'], order=order, color='#eb4c34');

ax.set_title('Accident Type Distribution - Accidents with Casualties')
plt.xticks(rotation=45);


# ## Seasonality of Accidents

# In[19]:


#Seasonal df
ses_df = df.set_index('date')
ses_df['count'] = 1

#Resample
ses_df = ses_df.resample('1m')[['count']].sum()


# In[20]:


#Plot
ax = ses_df.plot();

ax.set_title('Number of Accidents Throughout The Year');


# In[21]:


#Weekday
day_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['day_of_week'] = df['date'].dt.dayofweek.map(day_map)


# In[22]:


#Plot
ax = sns.countplot(df['day_of_week'], color='#46d4d1');

ax.set_title('Day Of Week');


# ## GeoLoc 

# In[86]:


#Constant
belgrade_loc = {'lat':'44.7866', 'long':'20.4489'}


# In[97]:


plt.figure(figsize=(ph,ph))

plt.scatter(df['long'], df['lat'], s=[5] * len(df), color='#32a89b');


# In[120]:


topl = df[(df['long'].between(20.3, 20.6)) & (df['lat'].between(44.6, 44.9))]
plt.figure(figsize=(ph,ph))

sns.kdeplot(topl['long'], topl['lat'], shade=True, shade_lowest=False, color='#32a89b');


# In[88]:


from gmplot import gmplot

from IPython.core.display import display, HTML
from IPython.display import IFrame


# In[95]:


#Create Heatmap
gmap = gmplot.GoogleMapPlotter(belgrade_loc['lat'],belgrade_loc['long'], zoom=10);

heatmap = gmap.heatmap(df['lat'], df['long'], radius=20)

hm_output = "accidents_heatmap.html"
gmap.draw(hm_output)


# In[108]:


#Display Map
IFrame(src=hm_output,width=700, height=600)


# ## Accident Descriptions

# In[140]:


df[df['description'].map(lambda x: 'bicikl' in x )]


# In[141]:


plt.figure(figsize=(pw, 2*pw))
df['description'].value_counts()[::-1].plot(kind='barh');


# In[ ]:




