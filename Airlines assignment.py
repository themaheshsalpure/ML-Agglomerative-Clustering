# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 22:42:38 2023

@author: Hp
"""

import pandas as pd
import matplotlib.pyplot as plt


"""
now import file from data sett and create a dataframe
"""

df=pd.read_excel("C:\\2-Dataset\\Assignment\\EastWestAirlines.xlsx")
df.describe()

df.info()

"""
This type of names can create confict hence to void this type of conflict 
we renamed this columns and removed special characters from names
"""
df=df.rename(columns={'ID#':'ID','Award?':'Award'})
df.ID
df.Award



#we have one column 'award' which really not useful we will drop it
df1=df.drop(["ID","Award"],axis=1)


df1.head(16)

#we know that there is scale difference among the columns, which we have to remove
#either by using normalization or standardization
#whenever there is mixed data apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x



"""
now apply this normalization function to df1 dataframe for first 15 rows
"""

df_norm=norm_func(df1.iloc[:,:])

"""
you can check the df_norm dataframe which is scaled between values from 0 to 1
you can apply describe function to new dataframe
"""

b=df_norm.describe()
b

"""
plot dendrogram first
now to create dendrogram we need to measure distance,
"""

#we have to import linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

#linkage function  gives us hirerarchicsl or algorithmic clustering
#ref the help for linkage
z=linkage(df_norm,method='complete',metric='euclidean')
plt.figure(figsize=(15,8));
plt.title("Hirerchical Clustering dendrogram");
plt.xlabel("Index");
plt.ylabel("Distance")
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

"""
applying agglomerative clustering choosing 3 as clusetrs
from dendrogram
whatever has been displayed in dendrogram is not clustering
it is just showing number of possible clusters
"""


from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df_norm)

#apply labels to the clusters

h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)


"""
assign thsi series to univ dataframe as column and name the column as "cluster
"""

df1['clust']=cluster_labels


"""
Using this syntax rearanged the columns to use them to ml algorithm
"""
df=df1.iloc[:,[3,1,2,4,5,6,7,8,9,10]]



df.iloc[:,2:].groupby(df.clust).mean()


"""
from the output cluster 2 has got highest Top 10
lowest accept ratio,best faculty ratio and highest expenses
highest graduates ratio
"""

"""
To save the information CLustered in our directory 
and os is used forgetting the information about the path where the file 
has been saved
"""
df.to_csv("Airlienes.csv",encoding='utf-8')
import os
os.getcwd()
