#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston 

boston = load_boston()

boston_df = pd.DataFrame(boston.data, columns = boston.feature_names)

boston_df.head()
boston_df['MEDV'] = boston.target
boston_df.head()
boston_df.isnull().sum()


# # corelation (-1,1) x----->y
# * if -1 inversaly proportional
# * if 1 directly prportinal
plt.figure(figsize=[20,10])
sns.heatmap(boston_df.corr(), annot = True)
plt.show()

##scatter Plot
plt.figure(figsize =[20,10])
plt.scatter(boston_df['LSTAT'],boston_df['MEDV'])
plt.show()

##MEDV Normalization Curve
plt.figure(figsize= [11,9])
sns.set_style('darkgrid')
sns.distplot(boston_df['MEDV'], color = 'Red')
plt.show()

#boston_df.drop(['CHAS'], axis = 1, inplace =True)
Y = boston_df[['MEDV']]

X = boston_df.drop(['MEDV'], axis = 1)

#X = boston_df.drop(['CHAS'], axis = 1)


X.columns


# # Min Max scalar

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(X)

scaled_data


# # Splitting Data
# 

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(scaled_data, Y, test_size=0.2, random_state = 3)
X_train.shape

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression(copy_X =True, fit_intercept=True, n_jobs=None, normalize= False)
lin_reg.fit(X_train, Y_train)


lin_reg.score(X_test, Y_test)


yhat =lin_reg.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error
r2_score(Y_test, yhat)

mean_squared_error(Y_test, yhat)

##coefficient and intercept

lin_reg.coef_


lin_reg.intercept_

