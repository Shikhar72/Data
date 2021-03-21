#!/usr/bin/env python
# coding: utf-8

# # Teen Market Segmentation Using K-means Clustering

# Interacting with friends on a social networking service (SNS) has become a rite of passage for teenagers around the world. The many millions of teenage consumers using such sites have attracted the attention of marketers struggling to find an edge in an increasingly competitive market. One way to gain this edge is to identify segments of teenagers who share similar tastes, so that clients can avoid targeting advertisements to teens with no interest in the product being sold. For instance, sporting apparel is likely to be a difficult sell to teens with no interest in sports.

# ## Dataset Information

# The dataset represents a random sample of 30,000 U.S. high school students who had profiles on a well-known SNS in 2006. To protect the users’ anonymity, the SNS will remain unnamed. The data was sampled evenly across four high school graduation years (2006 through 2009) representing the senior, junior, sophomore, and freshman classes at the time of data collection
# The dataset contatins 40 variables like: gender, age, friends, basketball, football, soccer, softball, volleyball,swimming, cute, sexy, kissed, sports, rock, god, church, bible, hair, mall, clothes, hollister, drugs etc whcih shows their interests. The final dataset indicates, for each person, how many times each word appeared in the person’s SNS profile

# ## Load Libraries

# In[1]:


# Importing Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns


# # Load Data

# In[2]:


pd.set_option('display.max_columns',None)
data = pd.read_csv(r"C:\Users\Archit\Desktop\Training\Imarticus\PGAA\PGAACombined\Clustering\snsdata.csv")
data.head()


# In[3]:


data.columns


# ## Summary Statistics

# ### Summary Statistics of Numerical Variables

# In[5]:


data.describe().T


# In[ ]:





# ### Summary Statistics of Categorical Variables

# In[10]:


data.describe(include='object')


# ## Treating Missing Values

# In[7]:


data.isnull().sum()


# A total of 5,086 records have missing ages. Also concerning is the fact that the minimum and maximum values seem to be unreasonable; it is unlikely that a 3 year old or a 106 year old is attending high school.

# Let's have a look at the number of male and female candidates in our dataset

# In[8]:


data['gender'].value_counts()


#  Let's have a look at the number of male, female and msiing values 

# In[9]:


data['gender'].value_counts(dropna = False)


# In[11]:


3171+435


# There are 22054 female, 5222 male teen students and 2724 missing values

# Now we are going to fill all the null values in gender column with “No Gender”

# In[12]:


data['gender'].fillna('not disclosed', inplace = True)


# In[13]:


data['gender'].isnull().sum()


# Also, the age cloumn has 5086 missing values. One way to deal with these missing values would be to fill the missing values with the average age of each graduation year

# In[69]:


data.groupby('gradyear')['age'].mean()


# From the above summary we can observe that the mean age differs by roughly one year per change in graduation year. This is not at all surprising, but a helpful finding for confirming our data is reasonable

# We now fill the missing values for each graduation year with the mean that we got as above

# We don't have any missing values in the 'age' column

# In[15]:


data.age.mean()


# In[16]:


data["age"].fillna(data.age.mean(), inplace=True)


# In[17]:


data.isnull().sum()


# From the above summary we can see that there are no missing values in the dataset

# ## Treating Outliers

# The original age range contains value from 3 - 106, which is unrealistic because student at age of 3 or 106 would not attend high school. A reasonable age range for people attending high school will be the age range between 13 to 21. The rest should be treated as outliers keeping the age of student going to high school in mind. Let's detect the outliers using a box plot below

# In[18]:


summy=data.describe(percentiles=[0.01,0.05, 0.1, 0.15,0.25,.5, .75, .9,.95,.99]).T


# In[20]:


summy.to_csv(r"C:\Users\Archit\Desktop\Training\Imarticus\PGAA\PGAACombined\Clustering\basic_sts.csv")


# In[72]:


data.columns.to_list()


# In[21]:


num_var=['age',
 'friends',
 'basketball',
 'football',
 'soccer',
 'softball',
 'volleyball',
 'swimming',
 'cheerleading',
 'baseball',
 'tennis',
 'sports',
 'cute',
 'sex',
 'sexy',
 'hot',
 'kissed',
 'dance',
 'band',
 'marching',
 'music',
 'rock',
 'god',
 'church',
 'jesus',
 'bible',
 'hair',
 'dress',
 'blonde',
 'mall',
 'shopping',
 'clothes',
 'hollister',
 'abercrombie',
 'die',
 'death',
 'drunk',
 'drugs']


# In[22]:


for col in data[num_var]:
    percentiles = data[col].quantile([0.01,0.99]).values
    data[col][data[col] <= percentiles[0]] = percentiles[0]
    data[col][data[col] >= percentiles[1]] = percentiles[1]


# In[26]:


data.describe(percentiles=[0.01,0.05, 0.1,.95,.99]).T


# In[23]:


sns.boxplot(data['age'])


# From the above summary we can observe that after treating the outliers the mininmum age is 13.719000 and the maximum age is 21.158000

# From the above boxplot we observe that there are no outliers in the age column

# ## Data Preprocessing

# A common practice employed prior to any analysis using distance calculations is to normalize or z-score standardize the features so that each utilizes the same range. By doing so, you can avoid a problem in which some features come to dominate solely because they have a larger range of values than the others.
# <br>The process of z-score standardization rescales features so that they have a mean of zero and a standard deviation of one. This transformation changes the interpretation of the data in a way that may be useful here. Specifically, if someone mentions Swimming three times on their profile, without additional information, we have no idea whether this implies they like Swimming more or less than their peers. On the other hand, if the z-score is three, we know that that they mentioned Swimming many more times than the average teenager.

# In[28]:


names = data.columns[5:40]
scaled_feature = data.copy()
names


# In[30]:


features = scaled_feature[names]


# In[31]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(features.values)


# In[32]:


features = scaler.transform(features.values)


# In[33]:


scaled_feature[names] = features
scaled_feature.head()


# ## Convert object variable to numeric 

# In[34]:


def gender_to_numeric(x):
    if x=='M':
        return 1
    if x=='F':
        return 2
    if x=='not disclosed':
        return 3


# In[35]:


scaled_feature['gender'] = scaled_feature['gender'].apply(gender_to_numeric)
scaled_feature['gender'].head()


# In[68]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=22)


# In[42]:


model = kmeans.fit(scaled_feature)


# In[46]:


model.labels_


# In[47]:


data["Cluster_3_sol"]=model.labels_


# In[48]:


data["Cluster_3_sol"].value_counts()


# In[55]:


mean_clus3=data.groupby(["Cluster_3_sol"]).mean(['age',
 'friends',
 'basketball',
 'football',
 'soccer',
 'softball',
 'volleyball',
 'swimming',
 'cheerleading',
 'baseball',
 'tennis',
 'sports',
 'cute',
 'sex',
 'sexy',
 'hot',
 'kissed',
 'dance',
 'band',
 'marching',
 'music',
 'rock',
 'god',
 'church',
 'jesus',
 'bible',
 'hair',
 'dress',
 'blonde',
 'mall',
 'shopping',
 'clothes',
 'hollister',
 'abercrombie',
 'die',
 'death',
 'drunk',
 'drugs']).T


# In[58]:


mean_clus3.to_excel("Clust.xlsx")


# In[57]:


import os 
os.chdir(r"C:\Users\Archit\Desktop\Training\Imarticus\PGAA\PGAACombined\Clustering")


# In[59]:


data.shape


# In[66]:


data.describe().T


# In[65]:


xyz["mean"].to_csv("mean.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# ## Elbow Method

# In[69]:


model.inertia_


# In[71]:


wcss=[]
for i in range(2,25):
    kmean_loop=KMeans(n_clusters=i, random_state=22)
    kmean_loop.fit(scaled_feature)
    wcss.append(kmean_loop.inertia_)


# In[72]:


plt.plot(range(2,25), wcss);
plt.plot()


# In[ ]:





# In[ ]:





# In[73]:


# Creating a funtion with KMeans to plot "The Elbow Curve"
wcss = []
for i in range(1,20):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(scaled_feature)
    wcss.append(kmeans.inertia_)
    print("Cluster", i, "Inertia", kmeans.inertia_)
plt.plot(range(1,20),wcss)
plt.title('The Elbow Curve')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') ##WCSS stands for total within-cluster sum of square
plt.show()


# The location of a bend (knee) in the plot is generally considered as an indicator of the appropriate number of clusters. Our Elbow point is around cluster size of 5. We will use k=5 to further interpret our clustering result

# In[79]:


kmeans5=KMeans(n_clusters=5)


# In[80]:


model5=kmeans5.fit(scaled_feature)


# In[81]:


data["Cluster_5_sol"]=model5.labels_


# In[82]:


data["Cluster_5_sol"].value_counts()


# In[83]:


mean_clus5=data.groupby(["Cluster_5_sol"]).mean(['age',
 'friends',
 'basketball',
 'football',
 'soccer',
 'softball',
 'volleyball',
 'swimming',
 'cheerleading',
 'baseball',
 'tennis',
 'sports',
 'cute',
 'sex',
 'sexy',
 'hot',
 'kissed',
 'dance',
 'band',
 'marching',
 'music',
 'rock',
 'god',
 'church',
 'jesus',
 'bible',
 'hair',
 'dress',
 'blonde',
 'mall',
 'shopping',
 'clothes',
 'hollister',
 'abercrombie',
 'die',
 'death',
 'drunk',
 'drugs']).T


# In[84]:


mean_clus5.to_csv("clust5.csv")


# In[86]:


new=pd.read_csv("new.csv")


# In[88]:


names = new.columns[5:40]
scaled_feature = new.copy()
names
features = scaled_feature[names]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_feature[names] = features
scaled_feature.head()


# In[90]:



def gender_to_numeric(x):
    if x=='M':
        return 1
    if x=='F':
        return 2
    if x=='not disclosed':
        return 3
scaled_feature['gender'] = scaled_feature['gender'].apply(gender_to_numeric)
scaled_feature['gender'].head()


# In[91]:



model5.predict(scaled_feature)


# ### Fit K-Means clustering for k=5

# In[ ]:





# In[74]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(scaled_feature)


# As a result of clustering, we have the clustering label. Let's put these labels back into the original numeric data frame.

# In[75]:


len(kmeans.labels_)


# In[76]:


data['cluster_3'] = kmeans.labels_


# In[77]:


data.head()


# ## Interpreting Clustering Results

# Let's see cluster sizes first

# In[78]:


plt.figure(figsize=(12,7))
axis = sns.barplot(x=np.arange(0,3,1),y=data.groupby(['cluster_3']).count()['age'].values)
x=axis.set_xlabel("Cluster Number")
x=axis.set_ylabel("Number of Students")


# From the above plot we can see that cluster 0 is the largest and cluster 1 has fewest teen students

# Let' see the number of students belonging to each cluster

# In[53]:


size_array = list(data.groupby(['cluster_3']).count()['age'].values)
size_array


# let's check the cluster statistics

# In[54]:


data.groupby(["cluster_3"]).mean()[["age", "basketball"]]


# In[58]:


data.groupby(["cluster_3"]).agg({"basketball":["mean", "count","sum"], "football":["sum"]})


# In[ ]:





# In[60]:


data.groupby(['cluster_3']).mean()[['basketball', 'football','soccer', 'softball','volleyball','swimming','cheerleading','baseball','tennis','sports','cute','sex','sexy','hot','kissed','dance','band','marching','music','rock','god','church','jesus','bible','hair','dress','blonde','mall','shopping','clothes','hollister','abercrombie','die', 'death','drunk','drugs']]


# The cluster center values shows each of the cluster centroids of the coordinates. The row referes to the five clusters,the numbers across each row indicates the cluster’s average value for the interest listed at the top of the column. Positive values are above the overall mean level.

# In[ ]:




