#!/usr/bin/env python
# coding: utf-8

# In[14]:


# %load ../../misc/utils/import.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import datetime

from pandas_ods_reader import read_ods
import os

from scipy.stats import binom_test

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


# In[15]:


#colors
aquam = '#46d4d1'
blue = '#45d4ff'
peach = '#f57542'
coral = '#eb4c34'
yellow = '#eef07d'


# In[16]:


#Contstants
input_dir = 'inputs'
output_dir = 'outputs/'


# ## Exploratory Data Anlysis of Traffic Accidents in Belgrade 

# In[17]:


#Read df
file_names =[input_dir + '/' +  x for x in np.array(os.listdir(input_dir))]
#Columns are missing
columns = ['id', 'date', 'long', 'lat', 'acc_outcome' ,'acc_type', 'description']

df = read_ods(file_names[0], 1, columns=columns)
for fn in file_names[1:]:
    df = pd.concat([df, read_ods(fn, 1, columns=columns)])
print("Number of accidents {}".format(len(df)))


# In[18]:


#Sample
print("Sample")
df.sample(5, random_state=23)


# In[19]:


#Add dummy
df['count'] = 1

#Correct falty data
df['long'] = df['long'].astype('str').map(lambda x: x.replace(',','.'))
df['lat'] = df['lat'].astype('str').map(lambda x: x.replace(',','.'))

#Expolicit type cast
df['long'] = df['long'].astype('float')
df['lat'] = df['lat'].astype('float')


# In[20]:


#To date-time
df['date'] = pd.to_datetime(df['date'])

#Sort by date
df = df.sort_values('date')

#Clip end
start_date = datetime.datetime(2015, 1, 1)
end_date = datetime.datetime(2019, 1, 1) # 2 28
df = df[df['date'].between(start_date, end_date)]

#Month
df['month']  = df['date'].dt.month_name()

#Day
df['day']  = df['date'].dt.day

#Cal Date
df['cal_date'] = df['month'] + "_" + df['day'].astype(str)


#Weekday
df['day_of_week'] = df['date'].dt.weekday_name

#Hour
df['hour']  = df['date'].dt.hour

print("Data from {} to {}".format(df['date'].min().date(), df['date'].max().date()))


# In[21]:


#Check Duplicates
dupl_ids = df[df.duplicated(subset=['id'])]['id']

print('Check duplicates')
df.set_index('id').sort_index().loc[dupl_ids].head(4)


# In[22]:


#Drop Duplictates
print("Before duplicates removal {}".format(len(df)))
df = df.drop_duplicates(subset=['id'])
print("After duplicates removal {}".format(len(df)))


# In[23]:


#Filter incorrect AC types
at_vc = df['acc_type'].value_counts()
filter_ac = at_vc[at_vc < 1000].index 

df = df[df['acc_type'].map(lambda x: not x in filter_ac)]

#Filter incorrect Lat and Long
df = df[(df['long'].between(-180,22)) & (df['lat'].between(-90,90))] #22 - 180 faulty data


print("Number of accidents after innitial filtering {}".format(len(df)))


# In[24]:


#Output for trans
def output_csv_for_trnas(df, output_dir, col):
   df[col].value_counts().reset_index()[['index']].to_csv(output_dir + col + '.tsv', index=False) 

output_csv_for_trnas(df, output_dir, 'acc_type')
output_csv_for_trnas(df, output_dir, 'acc_outcome')
output_csv_for_trnas(df, output_dir, 'description')


# In[25]:


#Translate
tts = ['acc_type', 'acc_outcome', 'description']

for tt in tts:
    print("Translated " + tt)
    tt_df = pd.read_csv(output_dir + tt + '.tsv')
    tr_df = pd.read_csv(output_dir + tt + '_trans.tsv')
    #Create map
    trans_map = dict(zip(tt_df.values.flatten(), tr_df.values.flatten()))
    #Translate
    df[tt] = df[tt].map(trans_map)


# ## Accidents Outcomes

# In[26]:


#Plot
order = df['acc_outcome'].value_counts().index

ax = sns.countplot(df['acc_outcome'], order=order, color=blue);

ax.set_title('Accident Outcomes Distribution')
plt.xticks(rotation=45);


# In[27]:


df['acc_outcome'].value_counts()


# ## Accident Types

# In[28]:


#Plot
order = df['acc_type'].value_counts().index

ax = sns.countplot(df['acc_type'], order=order, color=blue);

ax.set_title('Accident Type Distribution - All')
plt.xticks(rotation=45);

plt.gca().axvline(df['acc_type'].nunique() - 1, color = coral);


# In[29]:


#Plot
topl = df[df['acc_outcome'] == 'with injured']
order = df['acc_type'].value_counts().index

ax = sns.countplot(topl['acc_type'], order=order, color=yellow);

ax.set_title('Accident Type Distribution - Accidents with Injuries')
plt.xticks(rotation=45);

plt.gca().axvline(df['acc_type'].nunique() - 1, color = coral);


# In[30]:


#Plot
topl = df[df['acc_outcome'] == 'with death']
order = df['acc_type'].value_counts().index

ax = sns.countplot(topl['acc_type'], order=order, color=peach);

ax.set_title('Accident Type Distribution - Accidents with Death')
plt.xticks(rotation=45);

plt.gca().axvline(df['acc_type'].nunique() - 1, color = coral);


# **Red Line** marks **pedestrians**. <br/>
#     Although the accidents with pedestrians are the **least common**, they have the **highest death toll** as well as **high number of injuries**. 

# ## Accident Descriptions

# In[31]:


plt.figure(figsize=(pw, 2*pw))
df['description'].value_counts()[::-1].plot(kind='barh', color=blue);


# ##  Time Series - Trend and Seasonality Observations

# In[32]:


#Seasonal df
ts_df = df.set_index('date')

#Resample
ts_df = ts_df.resample('1m')[['count']].sum()

#Trend and Season
ts_df['trend'] = ts_df[['count']].rolling(12).mean()
ts_df['residual'] = ts_df['count'] - ts_df['trend']


# In[33]:


#Plot
ax = ts_df.plot(colormap='cool_r');

ax.set_title('Number of Accidents');


# **Notes**
# - We can observe that the number of accidents is **slightly rising** each year
# - There might be **missing data** for 2015 Nov - 2016 Jan
# - 2019 data looks **odd**
# - There is still **seasonality left in residual** (monthly seson.)

# In[34]:


#Month
ax = sns.countplot(df['month'], color=aquam);

ax.set_title('Month');


# In[35]:


#Month
order = ['Monday', 'Tuesday', 'Wednesday','Thursday', 'Friday', 'Saturday', 'Sunday']
ax = sns.countplot(df['day_of_week'], order=order, color=aquam);

ax.set_title('Week Day');


# In[36]:


#Month
ax = sns.countplot(df['hour'], color=aquam);

ax.set_title('Time Of Day');


# In[37]:


#Critical Dates
ax = df['cal_date'].value_counts()[:20].plot(kind='bar', color=peach);

ax.set_title("Top 20 Calendar Dates with Highest Number of Accidents ")
ax.set_ylabel('Num. of Accidents');

plt.xticks(rotation=45);


# ## GeoLoc 

# In[38]:


#Constant
belgrade_loc = {'lat':'44.7866', 'long':'20.4489'}


# In[39]:


plt.figure(figsize=(ph, ph))

plt.scatter(df['long'], df['lat'], s=[5] * len(df), color=blue);
plt.title("All Accident Types");


# In[40]:


sns.relplot(x="long", y="lat", hue="acc_outcome", alpha=.8, palette="autumn_r", height=ph, data=df);

plt.title("Accident Outcomes");


# In[41]:


sns.relplot(x="long", y="lat", hue="acc_type", alpha=.5, palette="cool", height=ph, data=df);

plt.title("Accident Types");


# In[42]:


from gmplot import gmplot

from IPython.core.display import display, HTML
from IPython.display import IFrame


# In[43]:


#Create Heatmap
gmap = gmplot.GoogleMapPlotter(belgrade_loc['lat'],belgrade_loc['long'], zoom=10);

heatmap = gmap.heatmap(df['lat'], df['long'], radius=20)

hm_output = output_dir + "accidents_heatmap.html"
gmap.draw(hm_output)


# In[45]:


#Create Scatter
gmap = gmplot.GoogleMapPlotter(belgrade_loc['lat'],belgrade_loc['long'], zoom=10);

heatmap = gmap.scatter(df['lat'], df['long'], peach,  size=10, marker=False)

hm_output = output_dir + "accidents_scatter.html"
gmap.draw(hm_output)


# In[210]:


#Display Map
#IFrame(src=hm_output, width=800, height=800)


# **HTML** Available in **outputs** folder

# ## Traffic Circle Slavija  - Case Study

# - **pre_rec** - 1 year period before reconstruction
# - **post_rec** - 1 year period after reconstruction

# In[211]:


#Slavija coordinates
lat_1, long_1 = 44.803230, 20.465422
lat_2, long_2 = 44.802029, 20.467424

#Is Slavija
df['is_slavija'] = df['lat'].between(lat_2,lat_1) & df['long'].between(long_1, long_2)


# In[212]:


#Reconstruction Dates
rec_start = datetime.datetime(2017, 5, 1)
rec_end = datetime.datetime(2017, 12, 1)

pre_rec_start = rec_start - datetime.timedelta(days=365)
pre_rec_end = rec_start

post_rec_start = rec_end
post_rec_end = rec_end + datetime.timedelta(days=365)

#Reconstruction Period
def get_period(date):
    if date >= pre_rec_start and date <= pre_rec_end:
        return "pre_rec"
    elif date >= post_rec_start and date <= post_rec_end:
        return "post_rec"
    else:
        return "other"

df['pre_post_rec'] = df['date'].map(get_period)


# In[213]:


#Slavija pre and post reconstruction
sl_df = df[df['is_slavija'] & (df['pre_post_rec'] != 'other') ]

pp_df = df[df['pre_post_rec'] != 'other']


# In[214]:


f, ax = plt.subplots(1, 2, figsize=(2*ph, ph))

#Overall
sns.countplot(pp_df['pre_post_rec'], ax=ax[0],  palette="winter")

ax[0].set_title("Overall Accidents - Belgrade");

#Overall
sns.countplot(sl_df['pre_post_rec'], ax=ax[1], palette="winter")

ax[1].set_title("Overall Accidents - Slavija");


# In[215]:


post_c = sl_df['pre_post_rec'].value_counts()['post_rec']
pre_c = sl_df['pre_post_rec'].value_counts()['pre_rec']
x  = pre_c 
n = post_c + pre_c

post_d = pp_df['pre_post_rec'].value_counts()['post_rec']
pre_d = pp_df['pre_post_rec'].value_counts()['pre_rec']
p = pre_d / (post_d + pre_d)
res = binom_test(x, n, p,  alternative='less')
print ('C-Test for increase in number of accidents {:.2f} p-val'.format(res))


# **p-val above** general acceptance threshold. 

# **Notes** 
# - We can observe increase in the number of accidents on Slavija after reconstruction (right hand side)
# - General rise in the number of accident in Belgrade contributes only partially (left hand side)

# In[216]:


sns.catplot(x="pre_post_rec", y='count', col="acc_outcome", data=sl_df, kind="bar", estimator=sum, palette="winter",);


# In[217]:


sns.catplot(x="pre_post_rec", y='count', col="acc_type", data=sl_df, kind="bar", estimator=sum, palette="winter");


# **Notes**
# 1. **Less** accidents with **injuries**
# 2. **Less** accidents with **pedestrians** - this contributes to 1.
# 3. **Increased** number of accidents with multiple vehicles 
# 4. There were **no accidents** on Slavija with **deadly outcomes** (within two years)

# In[194]:


#Create ts dataframe
topl = sl_df.copy()

#Shift Year
topl['date_shf'] = topl['date'].map(lambda x : x.replace(year=2019))

topl = topl.set_index('date_shf')

#Pre/Post
topl['pre_rec'] = (topl['pre_post_rec'] == 'pre_rec').astype(int)
topl['post_rec'] = (topl['pre_post_rec'] == 'post_rec').astype(int)

topl = topl[['pre_rec', 'post_rec']].resample('1m').sum()


# In[195]:


ax = topl.plot( cmap="winter");

ax.set_title("Pre/Post Reconstruction Distribution of Accidents Throughout the Year");


# In[196]:


topl = sl_df

sns.relplot(x="long", y="lat", hue="pre_post_rec", alpha=.8, palette="winter", height=ph, data=topl);

plt.title("Accidents on Slavija - Pre/Post Reconstruction");


# In[199]:


topl = sl_df[sl_df['acc_type'] == 'WITH PEDESTRIANS']

sns.relplot(x="long", y="lat", hue="pre_post_rec", alpha=.8, palette="winter", height=ph, data=topl);

plt.title("Accidents with Pedestrians on Slavija - Pre/Post Reconstruction");


# ## Parking Lot Obilicev Venac

# In[149]:


#OV coordinates
lat_1, long_1 = 44.816212, 20.456927
lat_2, long_2 = 44.815384, 20.457687

#Is Slavija
pl_df = df[df['lat'].between(lat_2,lat_1) & df['long'].between(long_1, long_2)]

print("Number of accidents on Obilicev Venac parking lot {}".format(len(pl_df)))


# In[150]:


topl = pl_df

sns.relplot(x="long", y="lat", hue="acc_type", alpha=.8, palette="cool", height=ph, data=topl);

plt.title("Accidents on Obilicev Venac - Accident Type");


# In[151]:


sns.countplot(pl_df['acc_type'], palette='cool');


# Almost all accidents are with **parked cars** or **a single car**.
