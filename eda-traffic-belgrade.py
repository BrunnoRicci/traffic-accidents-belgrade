#!/usr/bin/env python
# coding: utf-8

# In[80]:


# %load ../../misc/utils/import.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import datetime

from pandas_ods_reader import read_ods
import os

#Display Settings
pw = 16
ph = 9
matplotlib.rcParams['figure.figsize'] = (pw, ph)

sns.set_style("whitegrid")

#Pandas Dsiplay
pd.options.display.max_rows = 100
pd.options.display.max_columns = 200
pd.options.display.max_colwidth = 200

# import warnings
# warnings.filterwarnings('ignore')


# In[81]:


#colors
aquam = '#46d4d1'
blue = '#45d4ff'
peach = '#f57542'
coral = '#eb4c34'


# ## Exploratory Data Anlysis of Traffic Accidents in Belgrade 

# In[82]:


#Read df
input_dir = 'inputs'

file_names =[input_dir + '/' +  x for x in np.array(os.listdir(input_dir))]
#Columns are missing
columns = ['id', 'date', 'long', 'lat', 'acc_outcome' ,'acc_type', 'description']

df = read_ods(file_names[0], 1, columns=columns)
for fn in file_names[1:]:
    df = pd.concat([df, read_ods(fn, 1, columns=columns)])
print("Number of accidents {}".format(len(df)))


# In[83]:


#Sample
print("Sample")
df.sample(5, random_state=23)


# In[84]:


#Add dummy
df['count'] = 1

#Correct falty data
df['long'] = df['long'].astype('str').map(lambda x: x.replace(',','.'))
df['lat'] = df['lat'].astype('str').map(lambda x: x.replace(',','.'))

#Expolicit type cast
df['long'] = df['long'].astype('float')
df['lat'] = df['lat'].astype('float')


# In[85]:


#To date-time
df['date'] = pd.to_datetime(df['date'])

#Sort by date
df = df.sort_values('date')

#Clip end
start_date = datetime.datetime(2015, 1, 1)
end_date = datetime.datetime(2019, 2, 28)
df = df[df['date'].between(start_date, end_date)]

#Month
df['month']  = df['date'].dt.month_name()

#Weekday
df['day_of_week'] = df['date'].dt.weekday_name

#Hour
df['hour']  = df['date'].dt.hour

print("Data from {} to {}".format(df['date'].min().date(), df['date'].max().date()))


# In[86]:


#Check Duplicates
dupl_ids = df[df.duplicated(subset=['id'])]['id']

print('Check duplicates')
df.set_index('id').sort_index().loc[dupl_ids].head(4)


# In[87]:


#Drop Duplictates
print("Before duplicates removal {}".format(len(df)))
df = df.drop_duplicates(subset=['id'])
print("After duplicates removal {}".format(len(df)))


# In[88]:


#Filter incorrect AC types
at_vc = df['acc_type'].value_counts()
filter_ac = at_vc[at_vc < 1000].index 

df = df[df['acc_type'].map(lambda x: not x in filter_ac)]

#Filter incorrect Lat and Long
df = df[(df['long'].between(-180,22)) & (df['lat'].between(-90,90))] #22 - 180 faulty data


print("Number of accidents after innitial filtering {}".format(len(df)))


# ## Accidents Outcomes

# In[89]:


#Plot
order = df['acc_outcome'].value_counts().index

ax = sns.countplot(df['acc_outcome'], order=order, color=blue);

ax.set_title('Accident Outcomes Distribution')
plt.xticks(rotation=45);


# In[90]:


df['acc_outcome'].value_counts()


# In[91]:


#df[df['acc_outcome'] == 'Sa poginulim'].sample(5, random_state=23)


# ## Accident Types

# In[92]:


#Plot
order = df['acc_type'].value_counts().index

ax = sns.countplot(df['acc_type'], order=order, color=blue);

ax.set_title('Accident Type Distribution - All')
plt.xticks(rotation=45);

plt.gca().axvline(df['acc_type'].nunique() - 1, color = coral);


# In[93]:


#Plot
topl = df[df['acc_outcome'] == 'Sa poginulim']
order = df['acc_type'].value_counts().index

ax = sns.countplot(topl['acc_type'], order=order, color=peach);

ax.set_title('Accident Type Distribution - Accidents with Death')
plt.xticks(rotation=45);

plt.gca().axvline(df['acc_type'].nunique() - 1, color = coral);


# **Red Line** marks **pedestrians**. <br/>
#     Although the accidents with pedestrians are the **least common**, they have the **highest death toll**. 

# ## Accident Descriptions

# In[94]:


plt.figure(figsize=(pw, 2*pw))
df['description'].value_counts()[::-1].plot(kind='barh', color=blue);


# ##  Time Series - Trend and Seasonality Observations

# In[100]:


#Seasonal df
ts_df = df.set_index('date')

#Resample
ts_df = ts_df.resample('1m')[['count']].sum()

#Trend and Season
ts_df['trend'] = ts_df[['count']].rolling(12).mean()
ts_df['residual'] = ts_df['count'] - ts_df['trend']


# In[101]:


#Plot
ax = ts_df.plot();

ax.set_title('Number of Accidents');


# **Notes**
# - We can observe that the number of accidents is **slightly rising** each year
# - There might be **missing data** for 2015 Nov - 216 Jan
# - 2019 data looks **odd**

# In[102]:


#Month
ax = sns.countplot(df['month'], color=aquam);

ax.set_title('Month');


# In[103]:


#Month
order = ['Monday', 'Tuesday', 'Wednesday','Thursday', 'Friday', 'Saturday', 'Sunday']
ax = sns.countplot(df['day_of_week'], order=order, color=aquam);

ax.set_title('Week Day');


# In[104]:


#Month
ax = sns.countplot(df['hour'], color=aquam);

ax.set_title('Time Of Day');


# ## GeoLoc 

# In[105]:


#Constant
belgrade_loc = {'lat':'44.7866', 'long':'20.4489'}


# In[106]:


plt.figure(figsize=(ph, ph))

plt.scatter(df['long'], df['lat'], s=[5] * len(df), color=blue);
plt.title("All Accident Types");


# In[107]:


sns.relplot(x="long", y="lat", hue="acc_outcome", 
            sizes=(40, 400), alpha=.8, palette="autumn_r",
            height=ph, data=df);

plt.title("Accident Outcomes");


# In[108]:


sns.relplot(x="long", y="lat", hue="acc_type",
            sizes=(5, 5), alpha=.5,
            height=ph, data=df);

plt.title("Accident Types");


# In[76]:


from gmplot import gmplot

from IPython.core.display import display, HTML
from IPython.display import IFrame


# In[77]:


#Create Heatmap
gmap = gmplot.GoogleMapPlotter(belgrade_loc['lat'],belgrade_loc['long'], zoom=10);

heatmap = gmap.heatmap(df['lat'], df['long'], radius=20)

hm_output = "outputs/accidents_heatmap.html"
gmap.draw(hm_output)


# In[78]:


#Display Map
IFrame(src=hm_output, width=800, height=800)


# **HTML** Available in **outputs** folder
