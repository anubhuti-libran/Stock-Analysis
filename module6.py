# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 23:12:24 2020

@author: Anubhuti Singh
"""

#ProblemStatement1
import pandas as pd
import math

csv_file_list = ["30 stocks/ADANIPORTS.csv",
                 "30 stocks/ADANIPOWER.csv",
                 "30 stocks/AJANTPHARM.csv", 
                 "30 stocks/AMARAJABAT.csv", 
                 "30 stocks/APOLLOTYRE.csv", 
                 "30 stocks/ASHOKA.csv", 
                 "30 stocks/ASIANPAINT.csv", 
                 "30 stocks/AXISBANK.csv", 
                 "30 stocks/BAJAJELEC.csv", 
                 "30 stocks/BAJFINANCE.csv", 
                 "30 stocks/BERGEPAINT.csv", 
                 "30 stocks/BOMDYEING.csv", 
                 "30 stocks/BPCL.csv", 
                 "30 stocks/CASTROLIND.csv", 
                 "30 stocks/CENTURYPLY.csv", 
                 "30 stocks/CIPLA.csv", 
                 "30 stocks/CUMMINSIND.csv", 
                 "30 stocks/MINDTREE.csv", 
                 "30 stocks/DRREDDY.csv", 
                 "30 stocks/EICHERMOT.csv", 
                 "30 stocks/EXIDEIND.csv", 
                 "30 stocks/FORTIS.csv", 
                 "30 stocks/GAIL.csv", 
                 "30 stocks/GMRINFRA.csv", 
                 "30 stocks/GUJALKALI.csv", 
                 "30 stocks/LT.csv", 
                 "30 stocks/IDFC.csv", 
                 "30 stocks/RAYMOND.csv", 
                 "30 stocks/ITDC.csv", 
                 "30 stocks/JETAIRWAYS.csv"]

close_price = pd.DataFrame()
for filename in csv_file_list:
    data = pd.read_csv(filename)
    df = pd.DataFrame(data)
    close_price = close_price.append(df['Close Price'])
    Close_price = close_price.transpose()

print(Close_price)
Close_price.to_csv('csv check.csv')

Close_price.columns = ["ADANIPORTS", "ADANIPOWER","AJANTPHARM",  "AMARAJABAT",  "APOLLOTYRE",  "ASHOKA",  "ASIANPAINT",
                       "AXISBANK", "BAJAJELEC",  "BAJFINANCE", "BERGEPAINT", "BOMDYEING",  "BPCL", "CASTROLIND", "CENTURYPLY", 
                       "CIPLA", "CUMMINSIND", "MINDTREE", "DRREDDY", "EICHERMOT", "EXIDEIND", "FORTIS", "GAIL", "MRINFRA", 
                       "GUJALKALI", "LT", "IDFC", "RAYMOND", "ITDC", "JETAIRWAYS"]

#ProblemStatement2
returns = Close_price.pct_change().mean() * 252
volatility = Close_price.pct_change().std() * math.sqrt(252)

returns.columns = ["Returns"]
volatility.columns = ["Variance"]

#Concatenating the returns and variances into a single data-frame
ret_var = pd.concat([returns, volatility], axis = 1).dropna()
ret_var.columns = ["Returns","Volatiltity"]

#ProblemStatement3
from sklearn.cluster import KMeans
import  pylab as pl
X =  ret_var.values #Converting ret_var into nummpy array
sse = []
for k in range(2,15):
    
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(X)
    
    sse.append(kmeans.inertia_) #SSE for each n_clusters
pl.plot(range(2,15), sse)
pl.title("Elbow Curve")
pl.show()

kmeans = KMeans(n_clusters = 5).fit(X)
centroids = kmeans.cluster_centers_
pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
pl.show()
        
Company = pd.DataFrame(ret_var.index)
cluster_labels = pd.DataFrame(kmeans.labels_)
df = pd.concat([Company, cluster_labels],axis = 1)



