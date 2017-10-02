
# coding: utf-8

# In[1]:


#!/usr/bin/python2.7

import numpy as np
import pandas as pd
from WindPackages import download, pre_clean, clean, trainNN


# In[2]:


print "download data..."
download.download_data()


# In[3]:


print "pre-clean data..."

pre_clean.pre_clean_data()


# In[3]:


print "post-clean data..."

clean.post_clean_data()


# In[4]:


print "train with neural network from sklearn..."
trainNN.train_nn()


# In[5]:


print "trained with neural network from Hongshan..."
trainNN.train_nn2()

