#!/usr/bin/env python
# coding: utf-8

# # Author: Kodoori Akshay

# # Task-2 Unsupervised Learning
# From the given 'iris' dataset,predicting the optimum number of clusters and represent it visually

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


from sklearn.model_selection import train_test_split


# In[3]:


from sklearn.tree import DecisionTreeClassifier


# In[4]:


from sklearn import tree


# In[5]:


from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


# In[6]:


df=pd.read_csv("C:/Users/Akshay/Downloads/iris.csv")
df.head(10)


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df['Species'].nunique()


# In[10]:


df['Species'].value_counts()


# In[11]:


sns.pairplot(df)


# In[12]:


x=df.iloc[:,0:4].values
y=df.iloc[:,4]


# In[13]:


from sklearn.cluster import KMeans


# In[14]:


plt.figure(figsize=(12,9))
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++',random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("elbow_method")
plt.xlabel("number of clusters")
plt.ylabel("wcss")


# In[15]:


print("from the elbow method we can see number of clusters 3")


# In[16]:


kmeans=KMeans(n_clusters=3,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(x)


# In[17]:


y_kmeans


# In[18]:


kmeans.cluster_centers_


# In[19]:


#Visualising the clusters - On the first two columns
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
           s = 100, c = 'red', label = 'iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
           s = 100, c = 'blue', label = 'iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
           s = 100, c = 'green', label = 'iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
           s = 100, c = 'yellow', label = 'Centroids')

plt.legend()

