#!/usr/bin/env python
# coding: utf-8

# # Assignment 1- Analysis on Haberman dataset.

# READING HABERMAN DATA
# 

# In[10]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as numpy
hb=pd.read_csv("haberman.csv")


# Number of datapoints and features

# In[3]:


print(hb.shape)


# Column names of the haberman dataset

# In[4]:


print(hb.columns)


# observation:
# a)there are four features of this haberman data.
# b)Age,year,nodes,survival status.
# 

# Description about the dataset.

# a)I have searched about
# haberman data i have found that it is the survival status of the patients who undergone the surgery of breast cancer. 
# 
# b)Age:It must be the age when the patient undergone with the surgery. 
# 
# c)Year:Year in which the surgery happened. 
# 
# d)Nodes:These are the lymph nodes that act as a filter that catches and trap cancer cells before they reach other parts of the body.
# 
# e)Status:It is the survival status of a person which is 1 or 2 ,here 1 represents that the person have survived more than 5 years after the surgery and 2 represents that the person have survived less than 5 years.

# How many people survived after the surgery?

# In[5]:


hb["status"].value_counts()


# Observation
# 1.225 people survived more than 5 years after surgery and 81 survived less than 5 years.
# 
# 2.Haberman data set is  an imbalanced data set.

# # 2D SCATTER PLOT

# My focus here is to find the best feature from which we can easily analyse the whole data set.
# 
# In this stage i do not know which feature is important . 
# 
# It is clear that number of nodes is the deciding feature as it detect the cancer.
# 
# Age of the person is also important as compared to year in which the surgery happened.

# In[6]:


#2d scatter plot with colour coding using seaborn


# In[21]:


sns.set_style("whitegrid")
sns.FacetGrid(hb,hue="status",height=4)    .map(plt.scatter,"nodes","age")    .add_legend();
plt.show();


#  I am  able to distinguish the survival status of the patient due to colour.But I  cannot conclude from here ,as i cannot separate them entirely.
# 
# As there are 3 features from which i can conclude my classification so how can we select any feature from all so that i can get output with less error.
# 

# # Pair plots between features

# In[24]:


plt.close()
sns.set_style('whitegrid')
sns.pairplot(hb,hue="status",size=4)    .add_legend();
plt.show()


# observations:
# 1.Age and nodes are most useful features among all .
#   What i guess previously was  verified here that the age and nodes are important features.
# 
# note:
# when the features or dimensions of the data set is less it will be easy to understand through pair plots
# but it will be difficult to understand through pair plots when features are more.
#     
#     
#      

# # 1 d scatter plot to seprate the data

# I am  using only one feature here ie. nodes which is the most deciding feature that aperson will survive or not after surgery.

# In[38]:


import numpy as np
hb_success=hb.loc[hb["status"] ==1];
hb_fail=hb.loc[hb["status"] ==2];
plt.plot(hb_success["nodes"],
np.zeros_like(hb_success["nodes"]),"o")
plt.plot(hb_fail["nodes"],
np.zeros_like(hb_fail["nodes"]),"o")
plt.show()


# observation:
# 1.1D Scatter plot using data feature age and nodes.
# 2.As i have seen that data of patients who failed to survive are overlapping with successfully survived person, i can not conclude from here
# 

# # pdf 

# In[40]:



sns.FacetGrid(hb,hue="status",size=5)    .map(sns.distplot,"age")    .add_legend();
plt.title('PDF OF SURVIVAL STATUS USING FEATURE AGE')
plt.show()


# observation:
# here i have seen that  age from 30 to 77 successfully survived person  status and failed to survive status are same ie.overlapping 
# so using age we cannot predict or conclude.

# In[41]:


sns.FacetGrid(hb,hue="status",size=5)    .map(sns.distplot,"nodes")    .add_legend();
plt.title('PDF OF SURVIVAL STATUS USING FEATURE NODES')
plt.show()


# observation:
# Above i have drawn plots using both age and nodes individually and here i found that most important feature is nodes here to detect the survival status.
# 1.this is the best datapoint ,I cannot distinguish perfectly but among all data points this the best way i can distinguish.
# 2.here i can say that 
# if nodes<=0 & nodes<= 4
# then successful survival status is high
# elseif nodes>=4
# then patients fail to survive status is high

# # cdf
# 

# In[45]:


counts, bin_edges = np.histogram(hb_success["nodes"], bins=10,density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf);
plt.title("CDF OF SUCCESSFULLY SURVIVED PERSON USING  NODES")
plt.show();


# observation
# 1.as the nodes increases the survival status of the patients reduces.
# 2.86% of the patients successfully survived if the number of nodes<5.
# 3.100% of the patients are failed to survive if the number of nodes gets closer to 40 and so on.

# In[46]:


counts, bin_edges = np.histogram(hb_fail["nodes"], bins=10,density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf);
plt.title("CDF OF FAILED TO SURVIVE PERSON USING  NODES")
plt.show()


# observation:
# 1.58% of the people are able to survive  if the number of nodes is less than 4
# 2.but if the number of nodes gets closer to 40 the patients will fail to survive with 100 % chances.
# 

# # MEAN MEDIAN AND STD-DEV

# MEAN

# In[47]:


print(np.mean(hb_success["nodes"]))


# Is the outlier disturbs the mean?
# 

# In[48]:


print(np.mean(np.append(hb_success["nodes"],50)))


# not reallly,it affects the mean slightly.
# 

# In[49]:


print(np.mean(hb_fail["nodes"]))


# observation.
# 1.with an outlier mean value changes ,so we have to be aware of outliers .
# 2.mean of nodes of the patients who fail to survive is quite high to the mean of nodes of the patients who successfully survived.This conclude that to survive after surgery there must be less number of nodes.
# 

# STD-DEV

# In[52]:


print(np.std(hb_success["nodes"]))
print(np.std(hb_fail["nodes"]))


# Is the outlier disturbs the STD-DEV?

# In[53]:


print(np.std(np.append(hb_success["nodes"],50)))


# An oulier affects the std-dev more than the mean of the data.

# # MEDIAN,PERCENTILES AND QUANTILES

# In[54]:


print("medians:")
print(np.median(hb_success["nodes"]))
print(np.median(np.append(hb_success["nodes"],50)))
print(np.median(hb_fail["nodes"]))
print("\nQuantiles:")
print(np.percentile(hb_success["nodes"],np.arange(0,100,25)))
print(np.percentile(hb_fail["nodes"],np.arange(0,100,25)))
print("\n90th percentile:")
print(np.percentile(hb_success["nodes"],90))
print(np.percentile(hb_fail["nodes"],90))
from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(hb_success["nodes"]))
print(robust.mad(hb_fail["nodes"]))


# observation:
# 1.median of sucessfully survived having nodes is 0 and median of failed to survive having nodes is 4
# 
# 2.quantiles is showing that 50% of the people sucessfully survived having nodes is 0.
# 
# 3.quantiles is showing that 50% of the people failed to survive having nodes is under 4.
# 
# 4.90 percentile  shows that 90% of the people sucessfully survived having nodes is under 8.
# 
# 5.90 percentile  shows that 90% of the people failed to survive having nodes is 20.

# # BOX PLOT ,CONTOUR PLOT AND VIOLIN PLOT

# BOX PLOT

# In[57]:


sns.boxplot(x="status",y="nodes",data=hb)
plt.title('BOX PLOT SHOWING THE SURVIVAL STAUS OF PERSON USING NODES')
plt.show()


# observation
# 1.box plot containing 3 lines show quantiles that is 25,50,75 percentiles here i have found that 50% of the patients who successfully survived has number of nodes =0
# 
# 2.i have also found that 50% of the patients who failed to  survive has number of nodes near 4.

# VIOLIN PLOT
# 

# In[58]:


sns.violinplot(x="status",y="nodes",data=hb)
plt.title('VIOLIN PLOT SHOWING THE SURVIVAL STAUS OF PERSON USING NODES')
plt.show()


# observation
# 1.middle of the dot in violin plot shows 50 percentile.
# 
# 2.exactly the same observation as boxplot
# 
# here 50% of the patients who was successfully survived  was having nodes=0
# and 50% of the patients who was failed to  survive  was having nodes >=4
# 

# CONTOUR PLOT
# 

# In[59]:


sns.jointplot(x="age",y="nodes",data=hb_success,kind="kde")
plt.grid()
plt.title('CONTOUR PLOT SHOWING THE SURVIVAL STAUS OF PERSON USING NODES')
plt.show()


# observation: 
# contour plot shows the contour and intensity of the contours help us to analyse our data.
# 
# here i have found that  age between 48 to 63  and nodes having 0 to 3 are having high intensity ie.high chances of successfully survived. 

# # Conclusion
# 

# a)We can diagnose the  Breast Cancer using Haberman’s Data set by applying various data analysis techniques and using various Python libraries like seaborn,matplotlib,etc.
# 
# b)Among all the four features ,nodes is the most important feature.Age also complements the feature  nodes in respect of analysis of survival status of the person who undergone surgery.
# 
# c)Number of nodes must be less than 4 in order to have higher chances of surviving sfter surgery.
# 
# d)As analyzed,people who survived successfully having mean of nodes is approx 3, and 8 for viceversa.
# 
# e)People having nodes closer to 40 are very much likely fail to survive.
# 
# f)Age between 48 to 63 and nodes having 0 to 3 are having high intensity ie.high chances of successfully survived.

# # References
# 

# a)https://www.appliedaicourse.com
# 
# b)https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival for the information of data set.
