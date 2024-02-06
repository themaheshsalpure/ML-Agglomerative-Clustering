    # -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:14:52 2023

@author: ASUS
"""

import pandas as pd
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt



data = pd.read_excel("University_Clustering.xlsx")
data
data.head(11)


data.describe()
data.columns

data.info()
'''
dropping the column named State from data bcz of no use
'''
data.drop(['State'], axis = 1, inplace = True)
data.info()



'''
temp = data.drop(['Univ'], axis = 1)
temp.info()

'''

# creating the function for the normalization of the data
def norm_funct(i):
    x = (i - i.min())/(i.max()-i.min())
    return x


'''
passing all the colums in the function except Univ as it is not of int 
'''
norm_data = norm_funct(data.iloc[:,1:])
norm_data.info()

norm_data.describe()



from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch


z = linkage(norm_data, method = 'complete', metric= 'euclidean')
plt.figure(figsize = (15,8));
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel("Index")
plt.ylabel('Distance')


sch.dendrogram(z, leaf_rotation = 0, leaf_font_size = 10)
plt.show()






from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=3, linkage='complete', metric='euclidean').fit(norm_data)
h_complete.labels_


cluster_labels = pd.Series(h_complete.labels_)
cluster_labels
    








