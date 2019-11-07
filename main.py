#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


raw_df = pd.read_csv('data/SE_rents2018_train.csv', index_col=0)

raw_test_df = pd.read_csv('data/SE_rents2018_test1.csv', index_col=0)


# # Data Summarize

# In[3]:


raw_df.describe()


# In[4]:


raw_df.shape


# In[5]:


raw_df.info()


# In[6]:


raw_df['rent'].hist(bins=100)


# ### Seperate all features into continuous, categorical and binary features.
# 
# For those none relatived features, we have excluded them from the features grouping: 
# - addr_unit: no relationship
# - building_id: no relationship
# - addr_city: hard to encode
# - addr_zip: hard to encode
# - addr_street: hard to process
# - neighborhood: hard to encode
# - line: hard to encode
# - bin: no relationship
# - bbl: no relationshio
# - description: hard to build a NLP model
# - unit: no relationship
# 

# In[92]:


continuous_features =['bathrooms','bedrooms','size_sqft','floor_count','year_built','min_to_subway','floornumber', 'addr_lat', 'addr_lon']
categorical_features =['borough']
binary_features = ['has_doorman', 'has_elevator', 'has_fireplace', 'has_dishwasher','is_furnished', 'has_gym', 'allows_pets', 
                   'has_washer_dryer','has_garage', 'has_roofdeck', 'has_concierge', 'has_pool', 'has_garden',
                   'has_childrens_playroom', 'no_fee', ]



# In[8]:


unique_count = [] 
for feature in raw_df.columns:
  unique_count.append(raw_df[feature].nunique())
count_df = pd.DataFrame({'Feature':raw_df.columns,'unique count': unique_count})
count_df


# ### use pair coorelation for continuous features

# In[9]:


continuous_df = raw_df[continuous_features+['rent']]
continuous_df.corr()['rent'][:-1]


# ### Check coorelation for binary features
# 
# 

# In[11]:


raw_df[binary_features+['rent']].corr()['rent'][:-1]
coor_results= []

for feature in binary_features:
  df = raw_df.groupby([feature]).aggregate(['mean'])['rent']
  df[feature]= df.index
  coor_results.append(df.corr().iloc[0][1])
coor_df = pd.DataFrame({'Coorelation': coor_results,'Feature':binary_features})
coor_df


# As we can see in the correlation table, all binrary features highly affected the rents. When we build the models, we should include all binary features.

# ### Check coorelation for categorical features
# Need to do the binary first, then check the coorelation for categorical features, should be doen by group two

# In[ ]:





# In[ ]:





# In[ ]:





# # Data Cleaning

# ## Cleaning Training dataset
# ### Handling missing data
# In order to handle missing data in this dataset, we frist find and count all the null values.

# In[10]:


raw_df.isna().sum()


# As we can see from the result,there are missing data appearing on: 
# - addr_unit
# - bin 
# - year_built 
# - min_to_subway 
# - description 
# - neighborhood 
# - unit 
# - floornumber 
# - line 
# 
# Base on our data exploration, we can see that in this case, all features beside year_built,min_to_subway,neighborhood,and floornumber has not much impact to our final result, thus we don't need to worry about them.
# 
# Then, we will be dropping the rows which we don't have values for year_built, min_to_subway, neighborhood, and floornumber.

# In[16]:


# We will call the new df md_df

md_df = raw_df.loc[
    raw_df.year_built.notnull() &
    raw_df.min_to_subway.notnull() & 
    raw_df.neighborhood.notnull() & 
    raw_df.floornumber.notnull()
]

# Reminder:
# use mode to replace NAN value, compare both method when creating models
# md_df = raw_df.loc[
#     raw_df.year_built.notnull() &
#     raw_df.min_to_subway.notnull() & 
#     raw_df.neighborhood.notnull() & 
# ]

# md_df['floornumber'].fillna(md_df['floornumber'].mode()[0], inplace=True)


print("original shape of dataset:",raw_df.shape)
print("shape of dataset after handling missing data:",md_df.shape)


# ## remove outliers

# In[22]:


for feature in continuous_features:
    md_df.plot.scatter(feature, 'rent')


# In[78]:


md_df.loc[md_df['size_sqft']==0].shape


# ## drop size_sqrt = 0 for now, since there are 713 rows, might replace with mode when creating models

# In[17]:


def remove_outliers(md_df, feature, low_value, high_value):
    print(feature, ': ', md_df.shape)
    md_df = md_df[md_df[feature]>low_value]
    md_df = md_df[md_df[feature]<=high_value]
    md_df.reset_index(drop=True,inplace=True)
    print(feature, ': ', md_df.shape)
    return md_df

md_df = remove_outliers(md_df, 'rent', 0, 30000)
md_df = remove_outliers(md_df, 'bathrooms', 0, 12)
md_df = remove_outliers(md_df, 'size_sqft', 0, 10000)
md_df = remove_outliers(md_df, 'floor_count', 0, 80)
md_df = remove_outliers(md_df, 'year_built', 1700, 2019)
md_df = remove_outliers(md_df, 'min_to_subway', 0, 60)
md_df = remove_outliers(md_df, 'floornumber', 0, 60)

md_df['year_built'] = 2019 - md_df['year_built'].astype(int)


# ## encode categorical feature and drop useless features

# In[18]:


boroughs = np.array(md_df['borough'].value_counts().index)

for borough in boroughs:
    md_df[borough] = md_df['borough'].apply(lambda x : int(x == borough))

features_notNeed = ['addr_unit', 'building_id', 'created_at', 'addr_street', 'addr_city', 'addr_zip', 'bin', 'bbl', 'description',                     'neighborhood', 'unit', 'borough', 'line']

md_df = md_df.drop(features_notNeed, axis=1)
md_df.head(10).T


# ## Cleaning dataset test1
# ### Handle missing data
# We frist find and count all the null values for test1 dataset

# In[4]:


raw_test_df.isna().sum()


# we will be dropping the rows which we don't have values for year_built, min_to_subway, and floornumber, and then rename the dataframe as test_df

# In[5]:


test_df = raw_test_df.loc[
    raw_test_df.year_built.notnull() &
    raw_test_df.min_to_subway.notnull() & 
    raw_test_df.neighborhood.notnull() & 
    raw_test_df.floornumber.notnull()
]

print("original shape of dataset:",raw_test_df.shape)
print("shape of dataset after handling missing data:",test_df.shape)

