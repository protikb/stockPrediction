
# coding: utf-8

# In[9]:

from matplotlib import pyplot as plt
import numpy as np
from pandas import read_csv
import math
from numpy import *


# In[10]:

dataframe = read_csv('spaces.csv')
dataset = dataframe.values
dataset = dataset.astype('float32')


# In[16]:

dataframe1 = read_csv('data1.csv')
dataset1 = dataframe1.values
dataset1 = dataset1.astype('float32')


# In[18]:

plt.plot(dataset[0:100],'r')
plt.plot(dataset1[0:100],'b')
plt.show()


# In[ ]:



