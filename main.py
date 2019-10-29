#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


raw_df = pd.read_csv('https://grantmlong.com/data/SE_rents2018_train.csv', index_col=0)
raw_df.head(20)


# # Data Exploration
# 
# ### Summarize and Plot a Histogram for the Target Variable
# 
# Is there anything surprising or interesting about this data?

# In[9]:


raw_df.describe()


# In[12]:


raw_df.shape


# In[10]:


raw_df.info()


# In[17]:


raw_df['rent'].hist(bins=100)


# In[21]:


sns.pairplot(raw_df[['rent','bedrooms','bathrooms']])


# In[37]:


raw_df[raw_df.columns[1:]].corr()['rent'][:-1]


# ### Feature Exploration
# ##### If we wanted to try to create a model to price any given apartment, what variables might be the most important?
# * How many variables are at our disposal?
# * Which are binary? Categorical? Continuous? 
# * Which variable make most sense to use from an intuitive standpoint?
# * Identify which variable(s) has the highest correlation with rents. Do these make sense?
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Scatterplots
# * Create a scatterplot of `size_sqft`, `bathrooms`, and `floor`.  
# * Describe the relationship you see? Is it positive or negative? Linear? Non-linear? 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




