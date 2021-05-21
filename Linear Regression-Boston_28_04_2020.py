#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston 


# In[2]:


boston = load_boston()


# In[3]:


boston_df = pd.DataFrame(boston.data, columns = boston.feature_names)


# In[4]:


boston_df.head()


# In[5]:


boston_df['MEDV'] = boston.target


# In[6]:


boston_df.head()


# # Data Manipulation

# In[ ]:





# In[ ]:





# In[7]:


boston_df.isnull().sum()


# # corelation (-1,1) x----->y
# * if -1 inversaly proportional
# * if 1 directly prportinal

# In[8]:


plt.figure(figsize=[20,10])
sns.heatmap(boston_df.corr(), annot = True)
plt.show()


# In[18]:


##scatter Plot
plt.figure(figsize =[20,10])
plt.scatter(boston_df['LSTAT'],boston_df['MEDV'])
plt.show()


# In[19]:


##MEDV Normalization Curve
plt.figure(figsize= [11,9])
sns.set_style('darkgrid')
sns.distplot(boston_df['MEDV'], color = 'Red')
plt.show()


# In[ ]:





# In[11]:


#boston_df.drop(['CHAS'], axis = 1, inplace =True)
Y = boston_df[['MEDV']]


# In[58]:


X = boston_df.drop(['MEDV'], axis = 1)


# In[59]:


#X = boston_df.drop(['CHAS'], axis = 1)


# In[69]:


X.columns


# # Min Max scalar

# In[70]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(X)


# In[71]:


scaled_data


# # Splitting Data
# 

# In[72]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(scaled_data, Y, test_size=0.2, random_state = 3)
X_train.shape


# In[73]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression(copy_X =True, fit_intercept=True, n_jobs=None, normalize= False)
lin_reg.fit(X_train, Y_train)


# In[74]:


lin_reg.score(X_test, Y_test)


# In[75]:


yhat =lin_reg.predict(X_test)


# In[76]:


from sklearn.metrics import r2_score, mean_squared_error
r2_score(Y_test, yhat)


# In[77]:


mean_squared_error(Y_test, yhat)


# In[ ]:


##coefficient and intercept


# In[79]:


lin_reg.coef_


# In[80]:


lin_reg.intercept_


# In[ ]:




