# -*- coding: utf-8 -*-
"""
Created on Sat May  9 23:02:16 2020

@author: Anubhuti Singh
"""

import numpy as np
import pandas as pd

#Problem Statement 1
df = pd.read_csv("INFY.csv")
df.head(10)
df.tail(10)
df.describe()

df.info()
df.index.values
df.columns 
df.Series[df['Series']=='eq']='EQ'
df.drop( df[ df['Series'] != 'EQ' ].index , inplace=True)
df1=pd.DataFrame(df)

#Problem Statement 2
df1.iloc[-90:-1,8].max()
df1.iloc[-90:-1,8].min()
df1.iloc[-90:-1,8].mean()

#Problem Statement 3
df1.dtypes
df1['Date'] = df1['Date'].astype('datetime64[ns]')
print(df1['Date'].max()-df1['Date'].min())

#Problem Statement 4
df1['year'] = pd.DatetimeIndex(df1['Date']).year
df1['month'] = pd.DatetimeIndex(df1['Date']).month
df1

gp = df1.groupby(['month','year'])
gp.first()

def vwap(df1):
    q = df1['No. of Trades'].values
    p = df1['Close Price'].values
    return df1.assign(vwap=(p * q).sum() / q.cumsum())


df2 = gp.apply(vwap)
df2.head(20)


#Problem Statement 5

def avgprice(n):
    avg = df1['Close Price'].iloc[-n].mean()
    print(avg)
avgprice(7)
avgprice(14)
avgprice(30)
avgprice(90)
avgprice(365)

def pnl(n):
    change = df1['Close Price'].iloc[-n] - df1['Close Price'].iloc[-1]
    per  = (change*100)/df1['Close Price'].iloc[-n]
    if change < 0:
        print("Loss:", per,"%")
    else:
        print("Profit:", per,"%")
pnl(7)
pnl(14)
pnl(30)
pnl(90)
pnl(365)

#Problem Statement 6
df1 = df1.iloc[1:]
df1.head()

df1['Day_Perc_Change']=df1['Close Price'].pct_change()
df1.head(10)

#Problem Statement 7
ans=''
def f(row):
    global ans
    if row['Day_Perc_Change'] >-0.5 and row['Day_Perc_Change']<0.5:
        ans = "Slight or No change"        
    elif row['Day_Perc_Change'] >0.5 and row['Day_Perc_Change']<1:
        ans = "Slight positive"
    elif row['Day_Perc_Change'] >-1 and row['Day_Perc_Change']<-0.5:
        ans = "Slight negative"  
    elif row['Day_Perc_Change'] >1 and row['Day_Perc_Change']<3:
        ans = "Positive"
    elif row['Day_Perc_Change'] >-3 and row['Day_Perc_Change']<-1:
        ans = "Negative"
    elif row['Day_Perc_Change'] >3 and row['Day_Perc_Change']<7:
        ans = "Among top gainers"
    elif row['Day_Perc_Change'] >-7 and row['Day_Perc_Change']<-3:
        ans = "Among top losers"    
    elif row['Day_Perc_Change'] > 7:
        ans = "Bull run"
    elif row['Day_Perc_Change'] <- 7:
        ans = "Bear drop"
    else:
        ans='NULL'
    return ans

df1['Trend'] = df1.apply(f, axis=1)
df1.head()

#Problem Statement 8
df1.groupby('Trend', as_index=False)['Total Traded Quantity'].mean()
df1.groupby('Trend', as_index=False)['Total Traded Quantity'].median()

#Problem Statement 9
df1.to_csv('week2.csv')