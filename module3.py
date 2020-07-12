# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 23:16:43 2020

@author: Anubhuti Singh
"""


import pandas as pd
df = pd.read_csv("GOLD.csv")
df=pd.DataFrame(df)

df.head(10)
df.describe()
df.info()
df.index.values 
df.shape
df.columns.values 
df.isnull()

df['Vol.'] = df['Vol.'].astype(str)
print(df['Vol.'].dtype)
df['Vol.'].head(5)

df['Vol.'] = df['Vol.'].replace({'K': '000'}, regex=True).map(pd.eval).astype(float) 
df['Vol.'].head(5)

df['Change %'] = df['Change %'].astype(str)
df['Change %'] = df['Change %'].replace({'%': ''}, regex=True).map(pd.eval).astype(float) 
df['Change %'].head(10)
df.isnull().head(5)

x = df[['Open','High','Low','Change %']]
x.head(5)

import numpy as np
np.where(x.values >= np.finfo(np.float64).max)


#ProblemStatement1
y = df[['Pred']]
y
y.index[y.isnull().any(axis=1)]

yn = y.iloc[:411]
xn = x.iloc[:411]

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
xn_train, xn_test, yn_train, yn_test = train_test_split(xn, yn, test_size=0.25, random_state=0)
regressor = LinearRegression()  
regressor.fit(xn_train, yn_train) 
#training the algorithm

yn_pred = regressor.predict(xn_test)
yn_pred

from sklearn.metrics import r2_score
r2_score(yn_test, yn_pred)

xm = x.iloc[411:]
ym_pred = regressor.predict(xm)
ym_pred
print('Coefficients: \n', regressor.coef_)

#Coefficients: [[ 2.95434678e+00 -1.14684101e-01 -2.84769715e+00  2.38776694e+02]]

from scipy import stats
import matplotlib.pyplot as plt

def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

#plot predicted vs actual
plt.plot(yn_pred,yn_test,'o')
plt.xlabel('Predicted')#,color='white')
plt.ylabel('Actual')#,color='white')
plt.title('Predicted vs. Actual: Visual Linearity Test')#,color='white')
plt.tick_params(axis='x', colors='white')
plt.tick_params(axis='y', colors='white')
abline(1,0)
plt.show()

#this shows that the pred column and the x variables followed a linear relationship

import seaborn as sns
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize =(5,5))
ax.set(ylabel='residuals',xlabel='fitted values')
sns.residplot(x = yn_pred, y = yn_test, lowess=True, color="g")
#residual plot
#-------


#trying the same model with 'new' column
y = df[['new']]
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
y_pred
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
print('Coefficients: \n', regressor.coef_)
#Coefficients: [[ -1.01049588   1.37844371  -0.37010883 121.61416156]]
#----

def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

#plot predicted vs actual
plt.plot(yn_pred,yn_test,'o')
plt.xlabel('Predicted')#,color='white')
plt.ylabel('Actual')#,color='white')
plt.title('Predicted vs. Actual: Visual Linearity Test')#,color='white')
plt.tick_params(axis='x', colors='white')
plt.tick_params(axis='y', colors='white')
abline(1,0)
plt.show()
#this shows that the new column and the x variables followed a linear relationship

#residual plot
import seaborn as sns
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize =(5,5))
ax.set(ylabel='residuals',xlabel='fitted values')
sns.residplot(x = y_pred, y = y_test, lowess=True, color="g")



'''--------------------------------'''
#ProblemStatement2

jet = pd.read_csv('JETAIRWAYS.csv', parse_dates=True, index_col='Date')
nifty = pd.read_csv('Nifty50.csv', parse_dates=True, index_col='Date')

# joining the closing prices of the two datasets 
daily_prices = pd.concat([jet['Close Price'], nifty['Close']], axis=1)
daily_prices.columns = ['JETAIRWAYS', 'Nifty50']

# check the head of the dataframe
print(daily_prices.head())

# calculate daily returns
daily_returns = daily_prices.pct_change(1)
clean_daily_returns = daily_returns.dropna(axis=0)  # drop first missing row
print(clean_daily_returns.head())

# split dependent and independent variable
X = clean_daily_returns['Nifty50']
y = clean_daily_returns['JETAIRWAYS']

import statsmodels.api as sm
# Add a constant to the independent value
X1 = sm.add_constant(X)

# make regression model 
model = sm.OLS(y, X1)

# fit model and print results
results = model.fit()
print(results.summary())


slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

print(slope)
#beta value came out to be : 1.1386125578017423

#for monthly return


jet1 = pd.read_csv('JETAIRWAYS.csv', parse_dates=True, index_col='Date')
nifty1 = pd.read_csv('Nifty50.csv', parse_dates=True, index_col='Date')

# joining the closing prices of the two datasets 
monthly_prices = pd.concat([jet1['Close Price'], nifty1['Close']], axis=1)
monthly_prices.columns = ['JETAIRWAYS', 'Nifty50']

# check the head of the dataframe
print(monthly_prices.head())

# calculate monthly returns
monthly_returns = monthly_prices.pct_change(30)
clean_monthly_returns = monthly_returns.dropna(axis=0) 
last_months_returns = clean_monthly_returns.iloc[-90:] # drop first missing row
print(clean_monthly_returns.head())


# split dependent and independent variable
X = last_months_returns['Nifty50']
y = last_months_returns['JETAIRWAYS']

import statsmodels.api as sm
# Add a constant to the independent value
X1 = sm.add_constant(X)

# make regression model 
model = sm.OLS(y, X1)

# fit model and print results
results = model.fit()
print(results.summary())

from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

print(slope)
#beta value came out to be : 2.5837926589439135