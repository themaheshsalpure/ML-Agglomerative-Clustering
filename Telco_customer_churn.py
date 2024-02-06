# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 23:47:03 2023

@author: ASUS
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

data = pd.read_excel('Telco_customer_churn.xlsx')
data


data.describe()


data.info()


new_data = data.drop(['Customer ID','Quarter','Referred a Friend','Offer','Phone Service','Multiple Lines','Internet Service','Internet Type','Online Security','Online Backup','Device Protection Plan','Premium Tech Support','Streaming TV','Streaming Movies','Streaming Music','Unlimited Data','Contract','Paperless Billing','Payment Method'], axis = 1)
new_data


def norm_funct(i):
    a = (i - i.min())/(i.max()-i.min())
    return a

norm_data = norm_funct(new_data)
norm_data


z = linkage(norm_data, method = 'complete', metric='euclidean')
plt.figure(figsize = (70,100))
plt.title("EastWestAirlines Dat aClustering")
plt.xlabel('Index')
plt.ylabel('Distance')


sch.dendrogram(z, leaf_rotation = 0, leaf_font_size = 10)
plt.show()






"""


data sclicing alternative technique


"""

data

data.info()
new = data.iloc[:,[1,4,5,8,12,24,25,26,27,28,29]]
new


fn = norm_funct(new.iloc[:,:10])
fn.info()

y = linkage(fn, method = 'complete', metric = 'euclidean')
















