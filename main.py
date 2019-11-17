#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import csv
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

get_ipython().magic(u'matplotlib inline')


# In[63]:


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

# In[5]:


continuous_features =['bathrooms','bedrooms','size_sqft','floor_count','year_built','min_to_subway','floornumber' ]
caterigal_features =['addr_street','addr_city','addr_zip','neighborhood','borough','line' ]
binary_features = ['has_doorman', 'has_elevator', 'has_fireplace', 'has_dishwasher','is_furnished', 'has_gym', 'allows_pets', 
                   'has_washer_dryer','has_garage', 'has_roofdeck', 'has_concierge', 'has_pool', 'has_garden',
                   'has_childrens_playroom', 'no_fee', ]


# ## Import external dataset from Internal Revenue Service
#  - We will import the 2017 individual income Tax statistic dataset from IRS website(https://www.irs.gov/pub/irs-soi/17zpallagi.csv).
#  - We will expend a new feature: **average_income** based on zipcode to our raw dataset 

# In[6]:


raw_income_data=pd.read_csv('https://www.irs.gov/pub/irs-soi/17zpallagi.csv', index_col=0)
raw_income_data.columns


# In[9]:


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

# In[7]:


raw_income_data = raw_income_data.loc[raw_income_data['STATE']=='NY']
raw_income_data.rename(columns = {'N1':'total_returns', 'A02650':'total_income'}, inplace = True) 
raw_income_data = raw_income_data[['STATE','zipcode','agi_stub','total_returns', 'total_income']]
raw_income_data = raw_income_data.loc[raw_income_data['zipcode']<99999]
raw_income_data = raw_income_data.loc[raw_income_data['zipcode']>0]


# ### Function to calculate the average income by zip code
# Each zip code has 6 different sizes of adjusted gross income which means we have 6 different number of total returns and total income for one zip code.
# By using the np.where and sum function, we can obtain the sum of income and sum of returns for each zip code. The income of the original dataset was in thousands of dollar so we need to multiply the sum of income by 1000 and then find the average. Since some zip code was not in the original set, we ingore those average that is NaN and only write the meaningful averages to csv file for future use.

# - Calculate average income 
# - Export to ny_income_2017.csv for storage
# - For next time, no need to import the raw_income_dataset again

# In[8]:


average_income = pd.DataFrame({'addr_zip':[],'zip_average_income':[]})

def calculate_avg_income():
    global average_income
    for zipcode in range(10001, 14906):
        current_sum=np.where(raw_income_data['zipcode']==zipcode, raw_income_data['total_income'],0).sum()
        current_returns=np.where(raw_income_data['zipcode']==zipcode, raw_income_data['total_returns'],0).sum() 
        if(current_returns <=0 or current_sum<=0):
            continue
        avg_income=(current_sum*1000)/current_returns
        new_row={'addr_zip':zipcode,'zip_average_income':avg_income}
        average_income=average_income.append(new_row,ignore_index=True)           
calculate_avg_income()
average_income.head(5)


#  - We realize that the income dataset is missing all income data between zipcode 11239 - 11354, we will take an averge of zipcode income for 11239 and 11354 to replace any zipcode income in between 
#  - In our training and testing dataset, only the zipcode income 11249 is missing

# In[12]:


print(list(set(raw_df['addr_zip']) - set(average_income['addr_zip'])))
print(list(set(raw_test_df['addr_zip']) - set(average_income['addr_zip'])))


# **Insert a new row for zipcode income 11249 into the average_income dataframe**

# In[57]:


avg_income = (average_income.loc[(average_income['addr_zip']==11239)].iloc[0]['zip_average_income'] +
             average_income.loc[(average_income['addr_zip']==11354)].iloc[0]['zip_average_income'])/2
new_row = {'addr_zip':11249,'zip_average_income':avg_income}
average_income=average_income.append(new_row,ignore_index=True)  


# ### Merge the raw train and test1 dataset with the income dataset by addr_zip

# In[64]:


raw_test_df=raw_test_df.reset_index().merge(average_income, how="left",on='addr_zip').set_index('rental_id')
raw_df=raw_df.reset_index().merge(average_income, how="left",on='addr_zip').set_index('rental_id')


# ### Find zip_average_income and rent cooleration

# In[11]:


continuous_features.append('zip_average_income')


# In[12]:


continuous_df = raw_df[['zip_average_income','rent']]
continuous_df.corr()['rent'][:-1]


# **The correlation between zip_average_income and rent is 0.403558, it is good enough to consider as a important feature that might impact the rent**

# # Data Cleaning

# ## Cleaning Training dataset
# ### Handling missing data
# In order to handle missing data in this dataset, we frist find and count all the null values.

# In[17]:


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

# In[65]:


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


# ## Remove outliers

# In[19]:


for feature in continuous_features:
    md_df.plot.scatter(feature, 'rent')


# In[20]:


md_df.loc[md_df['size_sqft']==0].shape


# **drop size_sqrt = 0 for now, since there are 713 rows, might replace with mode when creating models**

# In[66]:


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


# ### Encode categorical feature and drop useless features

# In[68]:


boroughs = np.array(md_df['borough'].unique())

for borough in boroughs:
    md_df[borough] = md_df['borough'].apply(lambda x : int(x == borough))

features_notNeed = ['addr_unit', 'building_id', 'created_at', 'addr_street', 'addr_city', 'addr_zip', 'bin', 'bbl', 'description',                     'neighborhood', 'unit', 'borough', 'line']

md_df = md_df.drop(features_notNeed, axis=1)


# ### Use pair coorelation for continuous features

# In[23]:


continuous_df = md_df[continuous_features+['rent']]
continuous_df.corr()['rent'][:-1]


# ### Check coorelation for binary features

# In[24]:


md_df[binary_features+['rent']].corr()['rent'][:-1]
coor_results= []

for feature in binary_features:
  df = raw_df.groupby([feature]).aggregate(['mean'])['rent']
  df[feature]= df.index
  coor_results.append(df.corr().iloc[0][1])
coor_df = pd.DataFrame({'Coorelation': coor_results,'Feature':binary_features})
coor_df


# As we can see in the correlation table, all binrary features highly affected the rents. When we build the models, we should include all binary features.

# ### Cleaning dataset test1
#  Handle missing data
# We frist find and count all the null values for test1 dataset

# In[25]:


raw_test_df.isna().sum()


# **we will be dropping the rows which we don't have values for year_built, min_to_subway, and floornumber, and then rename the dataframe as test_df**

# In[70]:


test_df = raw_test_df.loc[
    raw_test_df.year_built.notnull() &
    raw_test_df.min_to_subway.notnull() & 
    raw_test_df.neighborhood.notnull() & 
    raw_test_df.floornumber.notnull()
]

print("original shape of dataset:",raw_test_df.shape)
print("shape of dataset after handling missing data:",test_df.shape)


# ### Remove outliers of test dataset

# In[27]:


for feature in continuous_features:
    test_df.plot.scatter(feature, 'rent')


# In[71]:


test_df = remove_outliers(test_df, 'bathrooms', 0, 12)
test_df = remove_outliers(test_df, 'bedrooms', 0, 12)
test_df = remove_outliers(test_df, 'size_sqft', 0, 10000)
test_df = remove_outliers(test_df, 'year_built', 1700, 2019)
test_df = remove_outliers(test_df, 'min_to_subway', 0, 60)
test_df = remove_outliers(test_df, 'floornumber', 0, 60)

test_df['year_built'] = 2019 - test_df['year_built'].astype(int)


# **Encode categorical feature and drop useless features from test df**

# In[72]:


boroughs = np.array(test_df['borough'].unique())

for borough in boroughs:
    test_df[borough] = test_df['borough'].apply(lambda x : int(x == borough))

features_notNeed = ['addr_unit', 'building_id', 'created_at', 'addr_street', 'addr_city', 'addr_zip', 'bin', 'bbl', 'description',                     'neighborhood', 'unit', 'borough', 'line']

test_df = test_df.drop(features_notNeed, axis=1)


# In[73]:


md_df = shuffle(md_df).reset_index(drop=True)
test_df = shuffle(test_df).reset_index(drop=True)


# In[94]:


features = list(md_df.columns)
features.remove('rent')
k_fold = KFold(n_splits=5)


# In[107]:


def get_cv_results(regressor):
    
    results = []
    for train, test in k_fold.split(md_df):
        regressor.fit(md_df.loc[train, features], md_df.loc[train, 'rent'])
        y_predicted = regressor.predict(md_df.loc[test, features])
        accuracy = mean_squared_error(md_df.loc[test, 'rent'], y_predicted)**0.5
        results.append(accuracy)

    return np.mean(results), np.std(results)


# In[112]:


rforest = RandomForestRegressor(
    #random_state=random_state, 
    max_depth=5,
    n_estimators=100
)

rforest.fit(md_df[features], md_df['rent'])


# In[116]:


for feature,score in zip(features,rforest.feature_importances_):
    print(feature, ' ', score)


# In[ ]:




