#!/usr/bin/env python
# coding: utf-8

# ## K-means Algoritmasi
# 
# K-Means Adımları
# 
# 1.Küme sayısı(k) değerini belirle
# 2.Orta noktaları belirle
# 3.Ilk kümeyi olustur verileri en yakın kümeleri ata
# 4.İterasyonlar ile kümeleri daha iyi hale getir

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Sklearn ile K-means Kümeleme yapabiliriz
from sklearn.datasets import load_iris
from sklearn import cluster 
# Kümeleme için gereki cluster


# In[8]:


iris = load_iris()
X, y = load_iris(return_X_y=True)


# In[13]:


iris.feature_names


# In[ ]:


## grafiğini ciziyoruz


# In[20]:


plt.scatter(X[:,0] ,X[:,1],c= y)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
# X'ler numpy dizisi 
# scatter dağilim grafiği


# In[27]:


kmeans = cluster.KMeans(n_clusters = 3,random_state= 42)
#uzaklığı ölçüyor


# In[28]:


#verileri uydurma 
kmeans.fit(X)


# In[29]:


kmeans.cluster_centers_.round(2)
#orta nokta


# In[30]:


kmeans.labels_


# In[32]:


plt.scatter(X[:,0],X[:,1],c = kmeans.labels_)


# In[ ]:




