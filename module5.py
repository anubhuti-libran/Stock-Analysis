# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:50:40 2020

@author: Anubhuti Singh
"""


#Problem Statement 1
import pandas as pd
import math
import numpy as np
df = pd.read_csv("JETAIRWAYS.csv")
df=pd.DataFrame(df)

stock_mean = (df['Close Price']-df['Open Price'])/df['Open Price'].mean()
stock_mean
stock_std = (df['Close Price']-df['Open Price'])/df['Open Price'].std()
stock_std 

annual_mean=stock_mean*252
annual_stddev=stock_std * math.sqrt(252)

#ProblemStatement2
df1 = pd.read_csv("RAYMOND.csv")
df2 = pd.read_csv("LEMONTREE.csv")
df3 = pd.read_csv("TATAPOWER.csv")
df4 = pd.read_csv("HDFC.csv")
df5 = pd.read_csv("TITAN.csv")




for stock_df in (df1,df2,df3,df4,df5): 
	stock_df['Normed Return'] = stock_df['Close Price'] /stock_df.iloc[0]['Close Price']

#checking the column 'normed return'
df1.head()

for stock_df, allo in zip((df1,df2,df3,df4,df5),[.2,.2,.2,.2,.2]):
	stock_df['Allocation'] = stock_df['Normed Return']*allo
    
df2.head()

for stock_df in (df1,df2,df3,df4,df5):
	stock_df['Position Value'] = stock_df['Allocation']*1000000
    
df3.head()

# create list of all position values
all_pos_vals = [df1['Position Value'], df2['Position Value'], df3['Position Value'], df4['Position Value'],df5['Position Value']]

# concatenate the list of position values
portfolio_val = pd.concat(all_pos_vals, axis=1)
# set the column names
portfolio_val.columns = ['RAYMOND','LEMONTREE','TATAPOWER','HDFC','TITAN']
# add a total portfolio column
portfolio_val['Total'] = portfolio_val.sum(axis=1)
portfolio_val.head()

# Daily Return
portfolio_val['Daily Return'] = portfolio_val['Total'].pct_change(1)
# average daily return
portfolio_val['Daily Return'].mean()
# standard deviation
portfolio_val['Daily Return'].std()
#annual return
portfolio_val['Annual Return '] = portfolio_val['Daily Return']*252
#covariance matrix
portfolio_val.cov()


#ProblemStatement 3
for dd in (df1,df2,df3,df4,df5):
    dd.drop(dd.columns.difference(['Close Price']), 1, inplace=True)
stocks = pd.concat([df1, df2, df3,df4,df5],axis=1)
stocks.columns = ['RAYMOND','LEMONTREE','TATAPOWER','HDFC','TITAN']
# arithmetic mean daily return
stocks.pct_change(1).mean()
# arithmetic daily return
stocks.pct_change(1).head()
# log daily return
log_return = np.log(stocks/stocks.shift(1))
print(stocks.columns)

weights = np.array(np.random.random(5))
print('Random Weights:')
print(weights)

print('Rebalance')
weights = weights/np.sum(weights)
print(weights)

# expected return
print('Expected Portfolio Return')
exp_ret = np.sum((log_return.mean()*weights)*252)
print(exp_ret)

# expected volatility
print('Expected Volatility')
exp_vol = np.sqrt(np.dot(weights.T,np.dot(log_return.cov()*252, weights)))
print(exp_vol)

# Sharpe Ratio
print('Sharpe Ratio')
SR = exp_ret/exp_vol
print(SR)

num_ports = 5000
all_weights = np.zeros((num_ports, len(stocks.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for n in range(num_ports): 
    # wesights 
    weights = np.array(np.random.random(5)) 
    weights = weights/np.sum(weights)  
	
    # save the weights
    all_weights[n,:] = weights
	
    # expected return 
    ret_arr[n] = np.sum((log_return.mean()*weights)*252)

    # expected volatility 
    vol_arr[n] = np.sqrt(np.dot(weights.T,np.dot(log_return.cov()*252, weights)))

    # Sharpe Ratio 
    sharpe_arr[n] = ret_arr[n]/vol_arr[n]
    


# plot the data
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

#ProblemStatment4
#maximum Sharpe Ratio we got:  
sharpe_arr.max()
#location of the maximum Sharpe Ratio and then get the allocation for that index
sharpe_arr.argmax()
#ans=2431 (answer will vary everytime)

all_weights[2431,:]
#location of lowest volatility

vol_arr.min()
vol_arr.argmin()
#ans=3225(ans will vary everytime)

max_sr_ret = ret_arr[3906]
min_sr_vol = vol_arr[351]
print (min_sr_vol)


#plot the data
plt.figure(figsize=(12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
# add a red dot for max_sr_vol & max_sr_ret
plt.scatter(min_sr_vol, min_sr_vol, c='red', s=500,marker=(5,1,0), edgecolors='black')
plt.scatter(max_sr_ret, max_sr_ret, c='green', s=500,marker=(5,1,0), edgecolors='black')

plt.show()
