#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# ### Load the dataset

# In[2]:


os.chdir("C:\\Users\\Admin\\Desktop\\Training\\Algo_working\\PCA")


# In[ ]:


df_house_price = pd.read_csv('house_price.csv')


# In[4]:


df_house_price.head()


# ###  Exploratory Data Analysis

# In[5]:


df_house_price.columns


# In[6]:


df_house_price.shape


# df_house_price has 81 columns (79 features + id and target SalePrice) and 1460 entries (number of rows or house sales)

# In[7]:


numerical_var = df_house_price.dtypes[df_house_price.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_var))

categorical_var = df_house_price.dtypes[df_house_price.dtypes == "object"]
print("Number of Categorical features: ", len(categorical_var))


# The main check points would be the correlation between the numeric variables and target variable with multicollinearity.

# In[8]:


#filter numeric column only 
data_num = df_house_price[numerical_var]

#calculating correlation among numeric variable 
corr_matrix = data_num.corr() 

#filter correlation values above 0.5
filter_corr = corr_matrix[corr_matrix > 0.5]

#plot correlation matrix
plt.figure(figsize=(20,12))
sns.heatmap(filter_corr,
            cmap='coolwarm',
            annot=True);


# Based on the above correlation matrix, correlation among the variables been observed. For example, 
# "SalePrice" are correlated with "OverallQual" , "YearBuilt", "YearRemodAdd", "TotalBsmtSF", "1stFlrSF", "GrLivArea", "FullBath", "TotRmsAbvGrd", "GarageCars", and "GarageArea". 
# 
# It also show the multicollinearity. For example: the correlation between GarageCars and GarageArea is very high (0.88)
# 
# 

# ### SalePrice
# 

# In[9]:


sns.distplot(df_house_price['SalePrice'])


# From the above result, We can notice that values of "SalePrice" are not a normal distribution. It is positively skewed.
# 
# A Few people have very expensive house

# ### Relationship with numerical variables
# 

# ### Overall Quality

# In[10]:


#scatter plot OverallQual/saleprice
data = pd.concat([df_house_price["SalePrice"], df_house_price["OverallQual"]],axis=1)
data.plot.scatter(x="OverallQual", y="SalePrice", ylim=(0,800000));


# From the above result, We can say that the price of the house will be when the overall quality of the house is high. We can notice that for the same quality different price. Why?.
# "SalePrice" is correlated with other variables and "OverallQual" might be correlated with other variables. 

# ### Year Built
# 

# In[11]:


#scatter plot YearBuilt/saleprice
data = pd.concat([df_house_price["SalePrice"], df_house_price["YearBuilt"]],axis=1)
data.plot.scatter(x="YearBuilt", y="SalePrice", ylim=(0,800000));


# From the above plot, we can observe that the price of a house is comparatively more than the house was built recently. We also can notice that there are some outlier which means even if the house is too old, the price of the house is moderately high due to influences of other variables.

# ### Ground Living Area

# In[12]:


#scatter plot grlivarea/saleprice
data = pd.concat([df_house_price["SalePrice"], df_house_price["GrLivArea"]],axis=1)
data.plot.scatter(x="GrLivArea", y="SalePrice", ylim=(0,800000));


# While the size of the ground living area is increasing, the price of the houses is increasing. But
# even if the ground living area is high, the price of the house is low and when the ground living area is high, the price of the house is too high. Why?. 
# 
# We know the influence of other variables affects the price change of house.
# 
# When "GrLivArea" of a house is highly correlated with other variables, the price of the house is increasing and when "GrLivArea" of a house is not highly correlated with other variables, the price of the house is decreasing.

# ### TotalBsmtSF

# In[13]:


data = pd.concat([df_house_price["SalePrice"],df_house_price["TotalBsmtSF"]], axis=1)
data.plot.scatter(x="TotalBsmtSF", y="SalePrice", ylim=(0,800000));


# We can notice that "SalePrice" and "TotalBsmtSF" is with a linear relationship. We can see the value of "SalePrice" is going straight when the value of "TotalBsmtSF" is 0. Total square feet of basement area of a house is high but the price of the house is low due to the influence of other variables.

# 
# ### Visualizing categorical variables with "SalePrice".
# 

# ### House Style

# In[14]:


df_house_price.boxplot(column="SalePrice",        # Column to plot
                 by= "HouseStyle",         # Column to split upon
                 figsize= (8,8))


# ### Foundation

# In[15]:


df_house_price.boxplot(column="SalePrice",        # Column to plot
                 by= "Foundation",         # Column to split upon
                 figsize= (8,8))


# From the above plot, we can observe that if a house has "Poured Concrete" foundation, the price of the house is higher than other house prices.

# ### Garage Quality

# In[16]:



df_house_price.boxplot(column="SalePrice",        # Column to plot
                 by= "GarageQual",         # Column to split upon
                 figsize= (8,8))


# From the above plot, we can observe that if a house has a good garage, the price of the house is higher than other house prices. Some houses have an excellent garage. So the price of this kind of house is higher than all.

# So far, we have compared some variables with the target variable. We observed that what is the variables impact on target variable based EDA. If we want to reduce dimension, we can take only impact variables. This is one kind of way. Now we will use PCA to reduce the dimension of this dataset.

# ### Before apply PCA we have to handle missing value.

# In[17]:


#missing data
total_missing_value = df_house_price.isnull().sum().sort_values(ascending=False)
percent_of_missign_value = (df_house_price.isnull().sum()/df_house_price.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_missing_value, percent_of_missign_value], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# From above plot, We can see that which variable is correlated with "SalePrice".

# ### Imputing missing values

# PooQC: Data description says NA means "No Pool". In this data set, it has huge ratio of missing value(99%) and majority of houses have no Pool at all in general.

# In[ ]:


df_house_price["PoolQC"] = df_house_price["PoolQC"].fillna("None")


# ### MiscFeature

# Data description says NA means "no misc feature"

# In[ ]:


df_house_price["MiscFeature"] = df_house_price["MiscFeature"].fillna("None")


# ### Alley

# Data description says NA means "no alley access"

# In[ ]:


df_house_price["Alley"] = df_house_price["Alley"].fillna("None")


# ### Fence

# This column has NA value means "no fence"

# In[ ]:


df_house_price["Fence"] = df_house_price["Fence"].fillna("None")


# ### FireplaceQu

# This column has NA means "no fireplace"

# In[ ]:


df_house_price["FireplaceQu"] = df_house_price["FireplaceQu"].fillna("None")


# ### LotFrontage

# The area of each street connected to the house property most likely have a similar area to other houses in its neighborhood. So we can fill in missing values by the median LotFrontage of the neighborhood.

# In[ ]:


df_house_price["LotFrontage"] = df_house_price.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))


# ### GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None
# 

# In[ ]:


for i in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    df_house_price[i] = df_house_price[i].fillna('None')


# ### GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)
# 

# In[ ]:


for i in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    df_house_price[i] = df_house_price[i].fillna(0)


# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement

# for i in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
#     df_house_price[i] = df_house_price[i].fillna(0)

# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical basement-related features, NaN means that there is no basement.

# In[ ]:


for i in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df_house_price[i] = df_house_price[i].fillna('None')


# MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.

# In[ ]:


df_house_price["MasVnrType"] = df_house_price["MasVnrType"].fillna("None")
df_house_price["MasVnrArea"] = df_house_price["MasVnrArea"].fillna(0)


# MSZoning (The general zoning classification) : 'RL' is by far the most common value. So we can fill in missing values with 'RL'

# In[ ]:


df_house_price['MSZoning'] = df_house_price['MSZoning'].fillna(df_house_price['MSZoning'].mode()[0])


# Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
# 

# In[ ]:


df_house_price = df_house_price.drop(['Utilities'], axis=1)


# Functional : data description says NA means typical

# In[ ]:


df_house_price["Functional"] = df_house_price["Functional"].fillna("Typ")


# Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value

# In[ ]:


df_house_price['Electrical'] = df_house_price['Electrical'].fillna(df_house_price['Electrical'].mode()[0])


# KitchenQual: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.

# In[ ]:


df_house_price['KitchenQual'] = df_house_price['KitchenQual'].fillna(df_house_price['KitchenQual'].mode()[0])


# Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string

# In[ ]:


df_house_price['Exterior1st'] = df_house_price['Exterior1st'].fillna(df_house_price['Exterior1st'].mode()[0])
df_house_price['Exterior2nd'] = df_house_price['Exterior2nd'].fillna(df_house_price['Exterior2nd'].mode()[0])


# SaleType : Fill in again with most frequent which is "WD"

# In[ ]:


df_house_price['SaleType'] = df_house_price['SaleType'].fillna(df_house_price['SaleType'].mode()[0])


# MSSubClass : Na most likely means No building class. We can replace missing values with None

# In[ ]:


df_house_price['MSSubClass'] = df_house_price['MSSubClass'].fillna("None")


# In[ ]:


categorical_var = df_house_price.dtypes[df_house_price.dtypes == "object"]


# Transforming some numerical variables that are really categorical

# In[37]:


df_house_price['OverallCond'].dtype


# If we take the variable "OverallCond" which represents rates the overall condition of the house. So value of this column is from 1 to 10.
# 
#        10	Very Excellent
#        9	Excellent
#        8	Very Good
#        7	Good
#        6	Above Average	
#        5	Average
#        4	Below Average	
#        3	Fair
#        2	Poor
#        1	Very Poor
#        
# The datatype of this column is in "int64", but it should be in categorical. 
# 
# So we handle like this column.

# In[ ]:


#Changing OverallCond into a categorical variable
df_house_price['OverallCond'] = df_house_price['OverallCond'].astype(str)


# In[ ]:


#MSSubClass=The building class
df_house_price['MSSubClass'] = df_house_price['MSSubClass'].apply(str)


# In[ ]:


#Year and month sold are transformed into categorical features.
df_house_price['YrSold'] = df_house_price['YrSold'].astype(str)
df_house_price['MoSold'] = df_house_price['MoSold'].astype(str)


# Label Encoding some categorical variables that may contain information in their ordering set

# In[ ]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')


# In[ ]:


# process columns, apply LabelEncoder to categorical features
for c in cols:
    label_ec = LabelEncoder() 
    label_ec.fit(list(df_house_price[c].values)) 
    df_house_price[c] = label_ec.transform(list(df_house_price[c].values))


# In[43]:


df_house_price = pd.get_dummies(df_house_price)
print(df_house_price.shape)


# In[44]:


df_house_price.head()


# Remove target variable

# In[ ]:


X = df_house_price.drop('SalePrice',axis=1)  


# 
# ### Standardizing input variables

# In[ ]:


from sklearn.preprocessing import StandardScaler  
import numpy as np

# standardized the dataset
sc_x = StandardScaler()    
X_std = sc_x.fit_transform(X)


# #PCA from scratch using python

# ### 1. Computing the mean vector
# find the mean for each column

# In[ ]:


import numpy as np
mean_vec = np.mean(X_std, axis=0)


# ### 2. Computing the Covariance Matrix
# find the covariance among variables
# 

# In[48]:


cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)


# ### 3. Computing eigenvectors and corresponding eigenvalues
# find eigenvalues and eigenvectors
# 

# In[49]:


eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# If eigen value is high for eigen vector that means the vector has a lot of variance. 

# ### 4. Sorting the eigenvectors by decreasing eigenvalues

# In[50]:


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [[np.abs(eig_vals[i]), eig_vecs[:,i]] for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


# All the eigen values are sorted in an ascending order.

# ### 5. Select components based on eigen values

# We have 221 components from which we have to select components based on engen value which has high value.
# 
# Here, filter out eigen values which has above 0.5

# In[51]:


pairs = np.array(eig_pairs)
count = 0
components = []
for i in range(len(eig_pairs)):
  if eig_pairs[i][0] >= 0.5:
    count += 1
    components.append(pairs[i])
    
print("Number of components: " + str(count))

  


# When we select eigen value as 0.5 or above 0.5, we get 138 components.
# 0.5 is not a threshold eigen value. We use this value to check the percentage of information the selected component carry. if the selected component gives less information, we select more components and this can be achieved by setting eigen value below 0.5.

# Let's check how much information the selected components contains.

# In[ ]:


# calculate Explained Variance
total = 0
ein = []
for i in range(len(components)):
  total += components[i][0]
  ein.append(components[i][0])
  
#divide eigen value by total eigen value and then multiple with 100 for the selected components   
var_exp = [(i / sum(eig_vals))*100 for i in sorted(ein, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


# In[53]:


cum_var_exp


# We can notice that 138 components has 93.5% information among all components.
# 
# When we select 138 components we lose 6.5% information.

# ### 6. Select components based on scree plot

# In[54]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#Make a random array and then make it positive-definite
num_vars = len(eig_vals)
num_obs = 9

fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(num_vars) + 1
plt.plot(sing_vals, eig_vals, 'ro-', linewidth=2)

plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.xlim(0,225)                
plt.show()


# Scree Plot shows eigen value for each component.
# starting components has high value and in middle , some componenet has higher than others.
# What can we do using scree plot?
# We can select components based scree plot. If we select 150 components based on scree plot, We have to check how much information has the selected components.

# The following plot to shows percentage of variance explained by each of the selected components.

# In[ ]:


# calculate Explained Variance
total = sum(eig_vals)
#divide eigen value by total eigen value and then multiple with 100 for the selected components   
var_exp = [(i / total)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


# In[56]:


plt.figure()
plt.scatter([150],[cum_var_exp[150]],color='g')
plt.plot(cum_var_exp)
plt.xlabel('Principal Component')
plt.ylabel('Variance (%)') #for each component
plt.title('Dataset Explained Variance')
plt.show()


# From the above plot, we can observe that the variance remains constant after the number of components reaches 150

# In[57]:


cum_var_exp[150]


# From the above result, we can see that the selected 150 components has 96% information. Here we lose 4% information. We have to select less component than original variable, at the same time, the selected component must contain as much as information. 

# ### 7. Deriving the new data set

# Finally, we select only 150 components.
# Filter out eigen vector of the selected components.

# In[ ]:


N = 221
M = 150

a = np.ndarray(shape = (N, 0))
for i in range(M):
    b = eig_pairs[i][1].reshape(N,1)
    a = np.hstack((a, b))


# Perform matrix calculation of original dataset with eigen vector of the selected components.

# In[ ]:


# Projection Onto the New Feature Space
Y = X_std.dot(a)


# In[60]:


Y.shape


# We reduced the number of columns from 221 to 150.

# In[61]:


Y


# # PCA using sklearn

# ### Loading PCA module from sklearn
# loading pca module from sklearn
# 

# In[ ]:


from sklearn.decomposition import PCA as sklearnPCA


# We have 221 columns. So first we choose 221 components

# In[ ]:


# create instance for pca
sklearn_pca = sklearnPCA(n_components=221)

# fit and transform the instance on datad
Y_sklearn = sklearn_pca.fit_transform(X_std)


# In[64]:


print(sklearn_pca.explained_variance_ratio_)


# Percentage of variance explained by each of the selected components.
# 

# ### Select components based on singular value instead of eigen value as we saw before.

# In[65]:



sklearn_pca.singular_values_


# In[66]:


count = 0
for i in range(len(sklearn_pca.singular_values_)):
    if sklearn_pca.singular_values_[i]>= 22.5:
        count += 1 

print("Number of components: " + str(count))


# When we select singular value as 22.5 , we get 138 components.
# We use this value to check the percentage of information the selected component carry. If the selected component gives less information, we select more components and this can be achieved by setting singular value below 22.5.

# In[67]:


cum_sum = np.cumsum(sklearn_pca.explained_variance_ratio_)
cum_sum[155]


# 155 components has 96.9% of information about dataset.

# ### Select components based on Scree Plot

# In[68]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#Make a random array and then make it positive-definite
num_vars = len(sklearn_pca.singular_values_)
num_obs = 9

fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(num_vars) + 1
plt.plot(sing_vals,sklearn_pca.singular_values_, 'ro-', linewidth=2)
 
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Singular value')
plt.xlim(0,221)                
plt.show()


# Scree Plot shows singular value for each component.
# starting components has high value.
# What can we do using scree plot? 
# 
# From the above plot, we can observe that 150 components has singular value above 22.5
# 
# We can select components based on scree plot. If we select 150 components based on scree plot, we have to check how much information does the selected components has.

# The following plot shows percentage of variance explained by each of the selected components.

# In[69]:


plt.figure()
cum_sum = np.cumsum(sklearn_pca.explained_variance_ratio_)
plt.scatter([150],[cum_sum[150]],color='r')
plt.plot(np.cumsum(sklearn_pca.explained_variance_ratio_))
plt.xlabel('Principal Component')
plt.ylabel('Variance (%)') #for each component
plt.title('Dataset Explained Variance')
plt.show()


# From the above plot, we can observe that the variance remains constant after the number of components reaches 150
# 

# In[70]:


cum_sum = np.cumsum(sklearn_pca.explained_variance_ratio_)
cum_sum[150]


# 150 components has 96% information about the dataset.

# Finaly we select 150 components 

# In[ ]:


# create instance for pca
sklearn_pca = sklearnPCA(n_components=150)
# fit and transform the instance on datad
Y_sklearn = sklearn_pca.fit_transform(X_std)


# ### Print the new dataset

# In[72]:


Y_sklearn


# We reduced the number of columns from 221 to 150.
