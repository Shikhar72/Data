#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import scipy.stats as stats
import pandas_profiling   #need to install using anaconda prompt (pip install pandas_profiling)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 10, 7.5
plt.rcParams['axes.grid'] = True

from matplotlib.backends.backend_pdf import PdfPages

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA   


# In[3]:


# reading data into dataframe
telco= pd.read_csv("telco_csv.csv")


# In[4]:


telco.head()


# In[5]:


telco.info()


# In[5]:


pandas_profiling.ProfileReport(telco)


# In[2]:


# !pip install autoviz


# In[6]:


from autoviz.AutoViz_Class import AutoViz_Class


# In[8]:


telco_new = pd.get_dummies(telco, columns=['region'], drop_first=True, prefix='region') #one hot encoding


# In[9]:


telco_new = pd.get_dummies(telco_new, columns=['custcat'], drop_first=True, prefix='cust_cat') #one hot encoding


# In[10]:


#Handling missings - Method2
def Missing_imputation(x):
    x = x.fillna(x.median())
    return x

telco_new=telco_new.apply(lambda x: Missing_imputation(x))


# In[11]:


#Handling Outliers - Method2
def outlier_capping(x):
    x = x.clip_upper(x.quantile(0.99))
    x = x.clip_lower(x.quantile(0.01))
    return x
telco_new=telco_new.apply(lambda x: outlier_capping(x))


# In[12]:


telco_new.columns


# In[13]:


telco_new.drop(['wireless', 'equip'],axis=1, inplace=True)


# In[14]:


telco_new.columns


# In[15]:


telco_new


# In[16]:


# variable reduction (feature selection/Feature engineering) - PCA - principle component analysis
sc = StandardScaler()


# In[19]:


std_model = sc.fit(telco_new)


# In[18]:


std_model


# In[20]:


telco_scaled = pd.DataFrame(std_model.transform(telco_new), columns = telco_new.columns)


# In[21]:


pca_model = PCA(n_components=31)


# In[22]:


pca_model = pca_model.fit(telco_scaled)


# In[23]:


pca_model.explained_variance_  #eigen values


# In[24]:


pca_model.explained_variance_ratio_


# In[26]:


np.cumsum(pca_model.explained_variance_ratio_)  #Eigen values


# In[27]:


pca_model = PCA(n_components=10)

pca_model = pca_model.fit(telco_scaled)


# In[28]:


PCs = pd.DataFrame(pca_model.transform(telco_scaled), columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10' ])


# In[29]:


PCs


# In[30]:


#variable reduction
Loadings =  pd.DataFrame((pca_model.components_.T * np.sqrt(pca_model.explained_variance_)).T,columns=telco_new.columns).T


# In[31]:


Loadings.column= ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10' ]


# In[33]:


Loadings.to_csv('loadings.csv')


# In[31]:


#PCA can be used for any type of business problem (regressin, classificaiton, segmentation)
selected_vars = ['tollmon',
'voice',
'employ',
'internet',
'multline',
'reside',
'region_2',
'income',
'retire',
'gender'
]


# In[32]:


#Build the segmentation using two ways 
#1. Using PC's 2. Using selected variables from each PC's

final_input_segmentation = telco_scaled[selected_vars]


# In[33]:


km_3 = KMeans(n_clusters=3, random_state=123)
km_3 = km_3.fit(final_input_segmentation)


# In[34]:


km_3.cluster_centers_


# In[35]:


km_3.labels_


# In[36]:


km_4 = KMeans(n_clusters=4, random_state=123).fit(final_input_segmentation)
km_5 = KMeans(n_clusters=5, random_state=123).fit(final_input_segmentation)
km_6 = KMeans(n_clusters=6, random_state=123).fit(final_input_segmentation)
km_7 = KMeans(n_clusters=7, random_state=123).fit(final_input_segmentation)
km_8 = KMeans(n_clusters=8, random_state=123).fit(final_input_segmentation)


# In[37]:


telco_new


# In[38]:


telco_new['cluster_3'] = km_3.labels_
telco_new['cluster_4'] = km_4.labels_
telco_new['cluster_5'] = km_5.labels_
telco_new['cluster_6'] = km_6.labels_
telco_new['cluster_7'] = km_7.labels_
telco_new['cluster_8'] = km_8.labels_


# In[39]:


telco_new

#Choosing best solution (optimal solution) - Identifying best value of K
#Metrics	Silhoutte coeficient	between -1 & 1	
#		Closer to 1, segmentation is good	
#		closer to -1, segmentation is bad	
#	Pseudo F-value		
#Profiling of segments			
#Best practices			
#	Segment distribution	4%-40%	
#	Strategy can be implementable or not		
			

# In[40]:


#Finding optimal solutions (finding best value of K)
silhouette_score(final_input_segmentation, telco_new.cluster_3)


# In[41]:


# calculate SC for K=3 through K=12
k_range = range(3, 9)
scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=123)
    km.fit(final_input_segmentation)
    scores.append(silhouette_score(final_input_segmentation, km.labels_))


# In[42]:


# plot the results
plt.plot(k_range, scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.grid(True)


# In[43]:


scores


# In[44]:


telco_new.cluster_6.value_counts()/sum(telco_new.cluster_3.value_counts())


# In[45]:


pd.Series.sort_index(telco_new.cluster_6.value_counts())


# In[46]:


#Profiling
size=pd.concat([pd.Series(telco_new.cluster_3.size), pd.Series.sort_index(telco_new.cluster_3.value_counts()), pd.Series.sort_index(telco_new.cluster_4.value_counts()),
           pd.Series.sort_index(telco_new.cluster_5.value_counts()), pd.Series.sort_index(telco_new.cluster_6.value_counts()),
           pd.Series.sort_index(telco_new.cluster_7.value_counts()), pd.Series.sort_index(telco_new.cluster_8.value_counts())])


# In[47]:


Seg_size=pd.DataFrame(size, columns=['Seg_size'])
Seg_Pct = pd.DataFrame(size/telco_new.cluster_3.size, columns=['Seg_Pct'])


# In[48]:


pd.concat([Seg_size.T, Seg_Pct.T], axis=0)


# In[49]:


telco_new


# In[50]:


# Mean value gives a good indication of the distribution of data. So we are finding mean value for each variable for each cluster
Profling_output = pd.concat([telco_new.apply(lambda x: x.mean()).T, telco_new.groupby('cluster_3').apply(lambda x: x.mean()).T, telco_new.groupby('cluster_4').apply(lambda x: x.mean()).T,
          telco_new.groupby('cluster_5').apply(lambda x: x.mean()).T, telco_new.groupby('cluster_6').apply(lambda x: x.mean()).T,
          telco_new.groupby('cluster_7').apply(lambda x: x.mean()).T, telco_new.groupby('cluster_8').apply(lambda x: x.mean()).T], axis=1)


# In[51]:


Profling_output


# In[52]:


Profling_output_final=pd.concat([Seg_size.T, Seg_Pct.T, Profling_output], axis=0)


# In[53]:


#Profling_output_final.columns = ['Seg_' + str(i) for i in Profling_output_final.columns]
Profling_output_final.columns = ['Overall', 'KM3_1', 'KM3_2', 'KM3_3',
                                'KM4_1', 'KM4_2', 'KM4_3', 'KM4_4',
                                'KM5_1', 'KM5_2', 'KM5_3', 'KM5_4', 'KM5_5',
                                'KM6_1', 'KM6_2', 'KM6_3', 'KM6_4', 'KM6_5','KM6_6',
                                'KM7_1', 'KM7_2', 'KM7_3', 'KM7_4', 'KM7_5','KM7_6','KM7_7',
                                'KM8_1', 'KM8_2', 'KM8_3', 'KM8_4', 'KM8_5','KM8_6','KM8_7','KM8_8',]


# In[54]:


Profling_output_final


# In[55]:


Profling_output_final.to_csv('Profling_output_final.csv')


# #Predicting segment for new data

# In[63]:


new_cust = pd.read_csv('Telco_new_cust.csv')


# In[64]:


new_cust = pd.get_dummies(new_cust, columns=['region'], drop_first=True, prefix='region') #one hot encoding

new_cust = pd.get_dummies(new_cust, columns=['custcat'], drop_first=True, prefix='cust_cat') #one hot encoding


# In[65]:


new_cust.drop(['wireless', 'equip'],axis=1, inplace=True)


# In[66]:


telco_new.columns


# In[67]:


new_cust.columns


# In[68]:


#std_model = StandardScaler()
#std_model.fit(telco_new)
std_model.transform(new_cust)


# In[72]:


telco_scaled1 = pd.DataFrame(std_model.transform(new_cust), columns = new_cust.columns)


# In[71]:


selected_vars = ['tollmon',
'voice',
'employ',
'internet',
'multline',
'reside',
'region_2',
'income',
'retire',
'gender'
]


# In[ ]:


input_segmentation = telco_scaled1[selected_vars]


# In[ ]:


km_4.predict(final_input_segmentation)


# In[ ]:


new_cust['segment'] = km_4.predict(final_input_segmentation)


# In[ ]:




