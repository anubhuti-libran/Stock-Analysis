# -*- coding: utf-8 -*-
"""
Created on Sun May 24 23:54:23 2020

@author: Anubhuti Singh
"""

#Problem Statement 1
import pandas as pd
df = pd.read_csv("week2.csv")
data=pd.DataFrame(df)
data['Date'].dtypes
data['Date'] = data['Date'].astype('datetime64[ns]')
import matplotlib.pyplot as plt
data.set_index('Date')['Close Price'].plot(figsize=(10,5)).grid();

#Problem Statement 2
a=data['Day_Perc_Change']
b=data['Date']
plt.figure(figsize=(20,8))
plt.stem(b,a) 

#Problem Statement 3
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
data = data.set_index('Date')
sns.set(style="darkgrid")
scaledvolume =  data["No. of Trades"] - data["No. of Trades"].min()
scaledvolume = scaledvolume/scaledvolume.max() * data.Day_Perc_Change.max()
fig, ax = plt.subplots(figsize=(12, 12))
ax.stem(data.index, data.Day_Perc_Change , 'b', markerfmt='bo', label='Percent Change')
ax.plot(data.index, scaledvolume, 'k', label='Volume')
ax.set_xlabel('Date')
plt.legend(loc=2)
plt.tight_layout()
plt.xticks(plt.xticks()[0], data.index.date, rotation=45)
plt.show()

#From the above graphs it can be concluded that between july 2018 and october 2018, there has been a great downfall.


#Problem Statement 4
df1=pd.DataFrame(df)
df1['Trend'].value_counts()
import pandas as pd
from matplotlib import pyplot as plt
df1 = df1[1:]
colors=['#ccff99']
df1.groupby(['Trend']).size().plot(kind='pie',stacked=False,figsize=(15,10),autopct='%1.1f%%',startangle=130,shadow=False,colors=colors)
plt.ylabel('Trend')
plt.xlabel("")
plt.show()
ng = df1.groupby(['Trend'])
print(ng.groups)
averg = ng['Total Traded Quantity'].mean()

med = ng['Total Traded Quantity'].median()
print( med)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(df['Trend'], averg)
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(df['Trend'], med)
plt.show()
#There only exists one type of 'trend' in the dataset. Hence, the graphs do not contribute much to analysis.



#Problem Statement 5
data['% Dly Qt to Traded Qty'].plot.hist(bins=5,figsize=(15,10))
plt.xlabel("Daily return %")
data.reset_index()
#Problem Statement 6
df2 = data.set_index(data['Unnamed: 0'])
stock_A = pd.DataFrame(df2.iloc[:5,1:])
stock_A.reset_index(drop=True, inplace=True)
stock_B = pd.DataFrame(df2.iloc[5:10,1:])
stock_B.reset_index(drop=True, inplace=True)
stock_C = pd.DataFrame(df2.iloc[10:15,1:])
stock_C.reset_index(drop=True, inplace=True)
stock_D = pd.DataFrame(df2.iloc[15:20,1:])
stock_D.reset_index(drop=True, inplace=True)
stock_E = pd.DataFrame(df2.iloc[20:25,1:])
stock_E.reset_index(drop=True, inplace=True)

close_price = pd.DataFrame(stock_A['Close Price'])
close_price['Stock A'] = stock_A['Close Price']
close_price['Stock B'] = stock_B['Close Price']
close_price['Stock C'] = stock_C['Close Price']
close_price['Stock D'] = stock_D['Close Price']
close_price['Stock E'] = stock_E['Close Price']
close_price.columns.values
close_price = close_price.iloc[:,1:]

perc_change=pd.DataFrame()


perc_change['stock_A']=close_price['Stock A'].iloc[0:].pct_change()
perc_change['stock_B']=close_price['Stock B'].iloc[0:].pct_change()
perc_change['stock_C']=close_price['Stock C'].iloc[0:].pct_change()
perc_change['stock_D']=close_price['Stock D'].iloc[0:].pct_change()
perc_change['stock_E']=close_price['Stock E'].iloc[0:].pct_change()

perc_change = perc_change.iloc[1:]
import seaborn as sns
corr = perc_change.corr()
print(corr)    

ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap="Purples",
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
        

#Problem Statement 7
a=data['Day_Perc_Change'].rolling(7).std()
plt.plot(a)

#Problem Statement 8
import numpy as np
data['Log_Ret'] = np.log(data['Close Price'] / data['Close Price'].shift(1))
# Compute Volatility using the pandas rolling standard deviation function
data['Volatility'] = data['Log_Ret'].rolling(window=7).std() * np.sqrt(7)
print(data.tail(15))
# Plot the NIFTY Price series and the Volatility
data[['Close Price', 'Volatility']].plot(subplots=True, color='blue',figsize=(8, 6))


plt.figure(figsize=(20,8))
plt.plot(a)
plt.plot(data['Volatility'])
plt.xlabel("Date")

#Problem Statment 9
short_window = 21
long_window = 34
signals = pd.DataFrame(index=data.index)
signals['signal'] = 0.0
signals['SMA21'] = data['Close Price'].rolling(window=short_window, min_periods=1, center=False).mean()
signals['SMA34'] = data['Close Price'].rolling(window=long_window, min_periods=1, center=False).mean()
signals['signal'][short_window:] = np.where(signals['SMA21'][short_window:] 
                                            > signals['SMA34'][short_window:], 1.0, 0.0)   
signals['positions'] = signals['signal'].diff()
print(signals)
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(111,  ylabel='Price in Rupees')
signals[['SMA21', 'SMA34']].plot(ax=ax1, lw=2.)

ax1.plot(signals.loc[signals.positions == 1.0].index, 
         signals.SMA21[signals.positions == 1.0],
         '^', markersize=10, color='m')
        
ax1.plot(signals.loc[signals.positions == -1.0].index, 
         signals.SMA21[signals.positions == -1.0],
         'v', markersize=10, color='k')

plt.show()

#Problem Statement 10
rolling_mean = data['Close Price'].rolling(14).mean()
rolling_std = data['Close Price'].rolling(14).std()
data['Rolling Mean'] = rolling_mean
data['Bollinger High'] = rolling_mean + (rolling_std * 2)
data['Bollinger Low'] = rolling_mean - (rolling_std * 2)
data[['Close Price','Bollinger High','Bollinger Low']].plot(figsize=(15,10))

data.to_csv('week3.csv')
