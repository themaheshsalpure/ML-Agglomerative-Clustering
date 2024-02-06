# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 23:02:49 2023

@author: ASUS
"""


"""


2.	Perform clustering for the crime data and identify the number of clusters            
formed and draw inferences. Refer to crime_data.csv dataset.


"""

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

data = pd.read_csv('crime_data.csv')
data

# 5 numbers summary of the data
data.describe()


# getting insight about the dataype of the each column from the column
data.info()



#removing the first object data type column from the data
data_new = data.drop(data.iloc[:,:1], axis = 1)
data_new.info()


# function for finding the normal values 
def norm_funct(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

op = norm_funct(data_new)
op



z = linkage(op, method = 'complete', metric = 'euclidean')
plt.figure(figsize=(50,60))
plt.title('Crime Data Clustering')                       # Giving title to the plot
plt.xlabel('Index')                                      # Giving x - label to plot
plt.ylabel('Distance')                                   # Giving y - label to plot


# plotting the Dendrogram of the data points acording to their clusters
sch.dendrogram(z,leaf_rotation=0, leaf_font_size=10)
plt.show()














