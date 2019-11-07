#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import csv
import warnings
from sklearn.linear_model import LinearRegression

get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:

raw_df = pd.read_csv('https://grantmlong.com/data/SE_rents2018_train.csv', index_col=0)
raw_test_df = pd.read_csv('https://grantmlong.com/data/SE_rents2018_test1.csv', index_col=0)
raw_df.head(20)
raw_df.columns


# ## Data Explore

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
# For those none relatived features as below, we have excluded them from the features grouping: 
# - addr_unit: no relationship
# - building_id: no relationship
# - addr_lat: hard to analyze latitude
# - addr_lon: hard to analyze longtitude
# - bin: no relationship
# - bbl: no relationshio
# - description: hard to build a NLP model
# - unit: no relationship
# 

# In[18]:


continuous_features =['bathrooms','bedrooms','size_sqft','floor_count','year_built','min_to_subway','floornumber' ]
caterigal_features =['addr_street','addr_city','addr_zip','neighborhood','borough','line' ]
binary_features = ['has_doorman', 'has_elevator', 'has_fireplace', 'has_dishwasher','is_furnished', 'has_gym', 'allows_pets', 
                   'has_washer_dryer','has_garage', 'has_roofdeck', 'has_concierge', 'has_pool', 'has_garden',
                   'has_childrens_playroom', 'no_fee', ]


# ### Use pair coorelation for continuous features

# In[9]:


continuous_df = raw_df[continuous_features+['rent']]
continuous_df.corr()['rent'][:-1]


# ### Create a scatterplot of continuous features.  

# In[10]:


sns.pairplot(data = raw_df,  y_vars=['rent'],x_vars=continuous_features)


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

# ## Import external dataset from Internal Revenue Service
#  - We will import the 2017 individual income Tax statistic dataset from IRS website(https://www.irs.gov/pub/irs-soi/17zpallagi.csv).
#  - We will expend a new feature: **average_income** based on zipcode to our raw dataset 

# In[11]:


raw_income_data=pd.read_csv('https://www.irs.gov/pub/irs-soi/17zpallagi.csv', index_col=0)
raw_income_data.columns


# In[12]:


raw_income_data.describe()


# ### Income Dataset Description 
# This dataset comes from the IRS website's 2017 ZIP Code Data (SOI) about Individual Income Tax Statistics.
# According to the documentation's overview,the Statistics of Income (SOI) Division’s ZIP code data is tabulated using individual income tax returns (Forms 1040) filed with the Internal Revenue Service (IRS) during the 12-month period, January 1, 2018 to December 31, 2018.
# The original dataset contains many income and Tax Items, we only keep the ones that are relevant: 
# - STATEFIPS:The State Federal Information Processing System (FIPS) code
# - STATE: The State associated with the ZIP code
# - ZIPCODE: 5-digit Zip code
# - agi_stub: Size of adjusted gross income
# - N1: Total number of returns
# - A02650: Number of returns with total income
# 
# Our goal is to find the average income of each zipcode.

# ### Clean the raw income data and rename feature

# In[37]:


raw_income_data = raw_income_data.loc[raw_income_data['STATE']=='NY']
raw_income_data.rename(columns = {'N1':'total_returns', 'A02650':'total_income'}, inplace = True) 
raw_income_data = raw_income_data[['STATE','zipcode','agi_stub','total_returns', 'total_income']]
raw_income_data = raw_income_data.loc[raw_income_data['zipcode']<99999]
raw_income_data = raw_income_data.loc[raw_income_data['zipcode']>0]
raw_income_data.isna().sum()


# ### Function to calculate the average income by zip code
# Each zip code has 6 different sizes of adjusted gross income which means we have 6 different number of total returns and total income for one zip code.
# By using the np.where and sum function, we can obtain the sum of income and sum of returns for each zip code. The income of the original dataset was in thousands of dollar so we need to multiply the sum of income by 1000 and then find the average. Since some zip code was not in the original set, we ingore those average that is NaN and only write the meaningful averages to csv file for future use.

# - Calculate average income 
# - Export to ny_income_2017.csv for storage
# - For next time, no need to import the raw_income_dataset again

# In[27]:


def calculate_avg_income():
    with open('data/ny_income_2017.csv', mode='w') as avg_file:
        thewriter = csv.writer(avg_file)
        thewriter.writerow(['addr_zip','zip_average_income'])
        for zipcode in range(10001, 14906):
            current_sum=np.where(raw_income_data['zipcode']==zipcode, raw_income_data['total_income'],0).sum()
            current_returns=np.where(raw_income_data['zipcode']==zipcode, raw_income_data['total_returns'],0).sum()  
            avg_income=(current_sum*1000)/current_returns
            if(avg_income>0):
                thewriter.writerow([zipcode,avg_income])
    


# In[28]:


warnings.filterwarnings('ignore')
calculate_avg_income()


# <b>Read the ny_income_2017 file</b>

# In[29]:


average_income=pd.read_csv("data/ny_income_2017.csv")
average_income.head(5)


# ### Merge the raw dataset and the income dataset by addr_zip

# In[30]:


raw_df=raw_df.reset_index().merge(average_income, how="left",on='addr_zip').set_index('rental_id')
raw_df.head(5).T


# In[31]:


raw_test_df=raw_test_df.reset_index().merge(average_income, how="left",on='addr_zip').set_index('rental_id')
raw_test_df.head(5).T


# ### Find zip_average_income and rent cooleration

# In[32]:


continuous_features.append('zip_average_income')

# In[34]:


continuous_df = raw_df[['zip_average_income','rent']]
continuous_df.corr()['rent'][:-1]


# **The correlation between zip_average_income and rent is 0.403558, it is good enough to consider as a important feature that might impact the rent**

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

# In[11]:


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

# In[12]:


def remove_outliers(md_df, feature, low_value, high_value):
    print(feature, ': ', md_df.shape)
    md_df = md_df[md_df[feature]>low_value]
    md_df = md_df[md_df[feature]<=high_value]
    md_df.reset_index(drop=True,inplace=True)
    print(feature, ': ', md_df.shape)
    return md_df

md_df = remove_outliers(md_df, 'bathrooms', 0, 12)
md_df = remove_outliers(md_df, 'size_sqft', 0, 10000)
md_df = remove_outliers(md_df, 'year_built', 1700, 2019)
md_df = remove_outliers(md_df, 'min_to_subway', 0, 60)
md_df = remove_outliers(md_df, 'floornumber', 0, 60)

md_df['year_built'] = 2019 - md_df['year_built'].astype(int)


# ## encode categorical feature and drop useless features

# In[18]:


boroughs = np.array(md_df['borough'].unique())

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


# ## remove outliers of test df

# In[8]:


for feature in continuous_features:
    test_df.plot.scatter(feature, 'rent')


# In[14]:


test_df = remove_outliers(test_df, 'bathrooms', 0, 12)
test_df = remove_outliers(test_df, 'bedrooms', 0, 12)
test_df = remove_outliers(test_df, 'size_sqft', 0, 10000)
test_df = remove_outliers(test_df, 'year_built', 1700, 2019)
test_df = remove_outliers(test_df, 'min_to_subway', 0, 60)
test_df = remove_outliers(test_df, 'floornumber', 0, 60)

test_df['year_built'] = 2019 - test_df['year_built'].astype(int)


# ## encode categorical feature and drop useless features from test df

# In[15]:


boroughs = np.array(test_df['borough'].unique())

for borough in boroughs:
    test_df[borough] = test_df['borough'].apply(lambda x : int(x == borough))

features_notNeed = ['addr_unit', 'building_id', 'created_at', 'addr_street', 'addr_city', 'addr_zip', 'bin', 'bbl', 'description',                     'neighborhood', 'unit', 'borough', 'line']

test_df = test_df.drop(features_notNeed, axis=1)
test_df.head(10).T


# In[ ]:




