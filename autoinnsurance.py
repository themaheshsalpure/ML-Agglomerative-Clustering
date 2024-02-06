# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 08:35:29 2023

@author: ASUS
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch




data = pd.read_csv('AutoInsurance.csv')
data


data.describe()


data.info()




#we know that there is scale difference among the columns, which we have to remove
#either by using normalization or standardization
#whenever there is mixed data apply normalization

def norm_funct(i):
    a = (i - i.min())/(i.max()-i.min())
    return a



#now apply this normalization function to df1 dataframe for first 13 rows

norm_data = norm_funct(data)
norm_data




z = linkage(norm_data, method = 'complete', metric='euclidean')
plt.figure(figsize = (70,100))
plt.title("EastWestAirlines Dat aClustering")
plt.xlabel('Index')
plt.ylabel('Distance')

#linkage function  gives us hirerarchicsl or algorithmic clustering
#ref the help for linkage




sch.dendrogram(z, leaf_rotation = 0, leaf_font_size = 10)
plt.show()
# plot dendrogram first
#now to create dendrogram we need to measure distance,



#applying agglomerative clustering choosing 3 as clusetrs from dendrogram

from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(norm_data)


#apply labels to the clusters
h_complete.labels_


#assign thsi series to univ dataframe as column and name the column as "cluster
cluster_labels = pd.Series(h_complete.labels_)


data['clust'] = cluster_labels
data


data.iloc[:,[1,2,3,4,5]]


data.iloc[:,1:].groupby(data.clust).mean()


data.to_csv("Airlines Assignment.csv", encoding = 'utf-8')
import os
os.getcwd()





