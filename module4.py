# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 23:21:33 2020

@author: Anubhuti Singh
"""


#Problem Statement 1
import pandas as pd
df = pd.read_csv("week3.csv")
df=df.fillna(df.mean())
#1.1
val = " "
def f(row):
    global val
    if row['Close Price'] < row['Bollinger Low']:
        val = "Buy"
    elif row['Bollinger Low'] < row['Close Price'] and row['Rolling Mean']:
        val = "Hold Buy/Liquidate Short"
    elif row['Rolling Mean'] < row['Close Price'] and row['Bollinger High']:
        val = "Hold Short/Liquidate Buy"  
    elif row ['Bollinger High'] < row['Close Price']:
        val = "Short"
    else:
        val='null'
    return val

df['Call'] = df.apply(f, axis=1)
df


#1.2
x = df[['Bollinger High', 'Rolling Mean', 'Bollinger Low', 'Close Price']]
y = df[['Call']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

lr = LogisticRegression()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
knn = KNeighborsClassifier()
def classification(model, x, y):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
   
    score = []
    model.fit(x_train,y_train)
    print(' score - ',model.score(x_test,y_test))
    score.append(model.score(x_test,y_test))
    y_rfc_pred = model.predict(x_test)
    print(y_rfc_pred)
    
#logistic regression
classification(lr,x,y)

#Decision Tree Classifier
classification(dtc,x,y)

#Random Forest Classifier
classification(rfc,x,y)

#Knn classifier
classification(knn,x,y)
df.to_csv('week5info.csv')
#1.3
df1 = pd.read_csv("JETAIRWAYS.csv")
df1=df1.fillna(df1.mean())

rolling_mean = df1['Close Price'].rolling(14).mean()
rolling_std = df1['Close Price'].rolling(14).std()
df1['Rolling Mean'] = rolling_mean
df1['Bollinger High'] = rolling_mean + (rolling_std * 2)
df1['Bollinger Low'] = rolling_mean - (rolling_std * 2)
df1
df1=df1.fillna(df1.mean())
val = " "
def f(row):
    global val
    if row['Close Price'] < row['Bollinger Low']:
        val = "Buy"
    elif row['Bollinger Low'] < row['Close Price'] and row['Rolling Mean']:
        val = "Hold Buy/Liquidate Short"
    elif row['Rolling Mean'] < row['Close Price'] and row['Bollinger High']:
        val = "Hold Short/Liquidate Buy"  
    elif row ['Bollinger High'] < row['Close Price']:
        val = "Short"
    else:
        val='null'
    return val

df1['Call'] = df1.apply(f, axis=1)
df1

x = df1[['Bollinger High', 'Rolling Mean', 'Bollinger Low', 'Close Price']]
y = df1[['Call']]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    
#logistic regression
classification(lr,x,y)

#Decision Tree Classifier
classification(dtc,x,y)

#Random Forest Classifier
classification(rfc,x,y)

#Knn classifier
classification(knn,x,y)


#ProblemStatement2

df1['open_close_perc']= (df1['Open Price']-df1['Close Price'])/df1['Open Price']*100
df1["high_low_perc"]=(df1['High Price']-df1['Low Price'])/df1['High Price']*100
df1["close_roll_mean"] = df1['Close Price'].pct_change().rolling(5).mean()
df1["close_roll_std"] = df1['Close Price'].pct_change().rolling(5).std()

#creating function to create column 'Action'
df1['Action']=""
for i in range(0,len(df1)-1):
    if df1['Close Price'].loc[i+1] > df1['Close Price'].loc[i]:
        df1['Action'].loc[i] = 1
    else:
        df1['Action'].loc[i] = -1
        
df1['Action'].loc[len(df1)-1] = -1

#defining x and y
x = df1[["close_roll_std","close_roll_mean","high_low_perc","open_close_perc"]]
y = df1[['Action']]
x=x.fillna(x.mean())
y.tail()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
x=x.fillna(x.mean())
#running classification models
#Random Forest Classifier
classification(rfc,x,y)
df1.tail()
#Plotting
import numpy as np
df1['daily_return']=df1['Close Price'].pct_change()

df1['cumulative_return'] = np.exp(np.log1p(df1['daily_return']).cumsum())
df1['cumulative_return'].plot(figsize=(15,10))

df1.to_csv('week5jet.csv')