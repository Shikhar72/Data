#!/usr/bin/env python
# coding: utf-8

# # Reading and Understanding Data

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Importing all required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


car_data=pd.read_excel("CarData.xlsx")


# In[4]:


car_data.head()


# In[5]:


car_data.info()


# In[6]:


car_data.describe(include='all')


# In[7]:


car_data.shape


# Number of variables are 16,
# where 15 are independent variable and 1 dependent variable(Target variable.)

# In[8]:


car_data.columns


# In[9]:


car_data.isnull().sum().sort_values(ascending=False)


# In[10]:


car_data.nunique()


# # Dropping and Cleaning

# In[11]:


car_data.rename(columns={'Engine Fuel Type':'Enginefueltype','Engine HP':'EngineHP',
                         'Engine Cylinders':'Enginecylinders','Transmission Type':'Transmissiontype',
                        'Number of Doors':'Numberofdoors','Market Category':'Marketcategory',
                        'Vehicle Size':'Vehiclesize','Vehicle Style':'Vehiclestyle',
                        'highway MPG':'highwayMPG','city mpg':'citympg'},inplace=True)


# In[12]:


car_data.columns


# Here i have read the data and found that the variables have Space between them,
# 
# hence i have to rename the data.

#  ## Missing value Treatment

# In[13]:


car_data.dropna(inplace=True)


# In[14]:


car_data.isnull().sum()


# We have dropped the missing values because i think
# there is enough data  we have for making the good model still.

# # Visualising the data

# In[15]:


plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('Car Price Distribution Plot')
sns.distplot(car_data.Price)

plt.subplot(1,2,2)
plt.title('Car Price Spread')
sns.boxplot(y=car_data.Price)

plt.show()


# By the spread of data it seems like it has outliers after 500000
# 
# And there is also right skewness in the price .

# # Outliers Treatment

# In[16]:


Outliers_Car=car_data.describe(percentiles=[.01,.05,.1,.2,.25,.5,.75,.90,.95,.99]).T


# In[17]:


Outliers_Car.to_csv("Outliers_Car.csv")


# By checking the Outliers_Car,I have found that prices are the only variable which has outliers
# considering only the numerical variables
# 

# ## Capping Technique for outlier treatment 

# In[18]:


car_data.isnull().sum().sort_values()


# In[19]:


#car_data["Price"] >= 315888]  == 315888
#car_data[car_data["highwayMPG"] >= 45] == 45

#car_data[car_data["citympg"] >= 43] == 43
#car_data[car_data["EngineHP"] >= 631.17] == 631.17


# In[20]:


car_data["Price"]=np.where(car_data["Price"]>=315888,315888,car_data["Price"])
car_data["highwayMPG"]=np.where(car_data["highwayMPG"]>=45,45,car_data["highwayMPG"])
car_data["citympg"]=np.where(car_data["citympg"]>=43,43,car_data["citympg"])
car_data["EngineHP"]=np.where(car_data["EngineHP"]>=631.17,631.17,car_data["EngineHP"])


# # Univariate Analysis

# ## Visualising Categorical Data

# In[21]:


plt.figure(figsize=(25,10))

plt.subplot(1,4,1)
plt1=car_data.Make.value_counts().plot(kind='bar',color=["r", "b","g","y"])
plt.title("Companies histogram")
plt1.set(xlabel="Company's Name",ylabel="Frequency of Company",)


plt.subplot(1,4,2)
plt2=car_data.Enginefueltype.value_counts().plot(kind='bar',color=["r", "b","g","y"])
plt.title("Engine fueltype")
plt2.set(xlabel="Engine fueltype",ylabel="Frequency")

plt.subplot(1,4,3)
plt3=car_data.Transmissiontype.value_counts().plot(kind='bar',color=["r", "b","g","y"])
plt.title("Transmission type")
plt3.set(xlabel="Transmission type",ylabel="Frequency")


plt.subplot(1,4,4)
plt4=car_data.Driven_Wheels.value_counts().plot(kind='bar',color=["r", "b","g","y"])
plt.title("Driven Wheels")
plt4.set(xlabel="Driven Wheels",ylabel="Frequency")

plt.show()


# Inference :
#     
#     1.Cheverolet seemed to be favored car company.
#     
#     2.Number of regular unleaded cars are more.
#     
#     3.Automatic Cars are more preferred.
#     
#     4.Front wheel are most driven.
#     

# In[22]:


plt.figure(figsize=(25,10))

plt.subplot(1,3,1)
plt1=car_data.Marketcategory.value_counts().plot(kind='bar',color=["r", "b","g","y"])
plt.title("Market category")
plt1.set(xlabel="Category's Name",ylabel="Frequency of Category")


plt.subplot(1,3,2)
plt2=car_data.Vehiclesize.value_counts().plot(kind='bar',color=["r", "b","g","y"])
plt.title("Vehicle size")
plt2.set(xlabel="Vehicle Size",ylabel="Frequency")

plt.subplot(1,3,3)
plt3=car_data.Vehiclestyle.value_counts().plot(kind='bar',color=["r", "b","g","y"])
plt.title("Vehicle style")
plt3.set(xlabel="Vehicle style",ylabel="Frequency")


plt.show()


# Inference :
# 
# 1.Crossover seemed to be favored in market.
# 
# 2.Midsize vehicle are more preferred in market.
# 
# 3.SUVs are  the vehicle style which is most manufactured by the company ,
# may be they are more in demand

# ## Visualising the Numerical variables.

# In[23]:


sns.distplot(car_data["Year"])
plt.show()


# Manufacture of cars over the years and its price distribution

# In[24]:


sns.distplot(car_data["EngineHP"])
plt.show()


# In[25]:


sns.distplot(car_data["Enginecylinders"])
plt.show()


# In[26]:


sns.distplot(car_data["Numberofdoors"])
plt.show()


# In[27]:


sns.distplot(car_data["highwayMPG"])
plt.show()


# In[28]:


sns.distplot(car_data["citympg"])
plt.show()


# In[29]:


sns.distplot(car_data["Popularity"])
plt.show()


# In[30]:


sns.distplot(car_data["Price"])
plt.show()


# Price is our target variable,and i can easily see that it is right skewed.

# In[31]:


#Trying to normalize the target variable
sns.distplot(np.square(car_data["Price"]))
plt.show()


# In[32]:


sns.distplot(np.log1p(car_data["Price"]))
plt.show()


# I will try both the target variable while training the model and see the differences

# # Bivariate Analysis

# ## Categorical to Numerical Variables(X~Y) 

# In[33]:


plt.figure(figsize=(20,8))
plt.title('Engine Type Histogram')
sns.countplot(car_data.Enginefueltype, palette=("Blues_d"))
plt.show()


plt.figure(figsize=(20,8))
plt.title('Engine Type vs Price')
sns.boxplot(x=car_data.Enginefueltype, y=car_data.Price, palette=("PuBuGn"))
plt.show()

df = pd.DataFrame(car_data.groupby(['Enginefueltype'])['Price'].mean().sort_values(ascending = False))
df.plot.bar(figsize=(8,6))
plt.title('Engine Type vs Average Price')
plt.show()


# Inference:
#     1.Regular unleaded fuel type is more in amount ie. they are most favoured.
#     
#     2.Flex-Fuel(premium unleaded required /E85) are most costly whereas regular unleaded 
#     fuel type are less costly hence they are most favored.
#     
#     

# In[34]:


plt.figure(figsize=(20,8))
plt.title('Transmission Type Histogram')
sns.countplot(car_data.Transmissiontype, palette=("Blues_d"))
plt.show()


plt.figure(figsize=(20,8))
plt.title('Transmission Type vs Price')
sns.boxplot(x=car_data.Transmissiontype, y=car_data.Price, palette=("PuBuGn"))
plt.show()

df = pd.DataFrame(car_data.groupby(['Transmissiontype'])['Price'].mean().sort_values(ascending = False))
df.plot.bar(figsize=(8,6))
plt.title('Transmission Type vs Average Price')
plt.show()


# Inference:
#     1.Automatic cars are more available in market now.
#     
#     2.Automated_Manual cars have more cost.

# In[35]:


plt.figure(figsize=(20,8))
plt.title('Driven_wheels Histogram')
sns.countplot(car_data.Driven_Wheels, palette=("Blues_d"))
plt.show()


plt.figure(figsize=(20,8))
plt.title('Driven_wheels vs Price')
sns.boxplot(x=car_data.Driven_Wheels, y=car_data.Price, palette=("PuBuGn"))
plt.show()

df = pd.DataFrame(car_data.groupby(['Driven_Wheels'])['Price'].mean().sort_values(ascending = False))
df.plot.bar(figsize=(8,6))
plt.title('Driven_Wheels vs Average Price')
plt.show()


# Inference:
#     1.Front wheel drive cars are most manufactured they are in high demand
#     
#     2.Rear wheel drive cars are most costly.

# In[36]:


plt.figure(figsize=(20,8))
plt.title('Market category Histogram')
sns.countplot(car_data.Marketcategory, palette=("Blues_d"))
plt.show()


plt.figure(figsize=(20,8))
plt.title('Market category vs Price')
sns.boxplot(x=car_data.Marketcategory, y=car_data.Price, palette=("PuBuGn"))
plt.show()

df = pd.DataFrame(car_data.groupby(['Marketcategory'])['Price'].mean().sort_values(ascending = False))
df.plot.bar(figsize=(8,6))
plt.title('Market category vs Average Price')
plt.show()


# Inference:
#     1.Exotic Factory tuner cars having luxury performance has highest price.
#     
#     2.Hatchback,Factory tuner cars having luxury performance has highest price.

# In[37]:


plt.figure(figsize=(20,8))
plt.title('Vehicle Size Histogram')
sns.countplot(car_data.Vehiclesize, palette=("Blues_d"))
plt.show()


plt.figure(figsize=(20,8))
plt.title('Vehicle Size vs Price')
sns.boxplot(x=car_data.Vehiclesize, y=car_data.Price, palette=("PuBuGn"))
plt.show()

df = pd.DataFrame(car_data.groupby(['Vehiclesize'])['Price'].mean().sort_values(ascending = False))
df.plot.bar(figsize=(8,6))
plt.title('Vehicle Size vs Average Price')
plt.show()


# Inference:
#     1.Mid size vehicles are more in the market.
#     
#     2.Large size vehicles are costlier

# In[38]:


plt.figure(figsize=(20,8))
plt.title('Vehicle Style Histogram')
sns.countplot(car_data.Vehiclestyle, palette=("Blues_d"))
plt.show()


plt.figure(figsize=(20,8))
plt.title('Vehicle Style vs Price')
sns.boxplot(x=car_data.Vehiclestyle, y=car_data.Price, palette=("PuBuGn"))
plt.show()

df = pd.DataFrame(car_data.groupby(['Vehiclestyle'])['Price'].mean().sort_values(ascending = False))
df.plot.bar(figsize=(8,6))
plt.title('Vehicle Style vs Average Price')
plt.show()


# Inference:
#     1.4dr SUV are most preferrable vehicle style
#     
#     2.Convertible cars are having highest price while 2dr hatchback is having  least cost 

# ## Numerical to Numerical variable

# In[40]:


sns.lmplot(x="Year", y="Price", data=car_data)


# In[41]:


car_data.columns


# In[42]:


sns.lmplot(x="EngineHP", y="Price", data=car_data)


# In[43]:


sns.lmplot(x="Enginecylinders", y="Price", data=car_data)


# In[48]:


sns.lmplot(x="Numberofdoors", y="Price", data=car_data)


# Inference:
# 
# 1. Number of doors has no effects on price of the car.

# In[45]:


sns.lmplot(x="highwayMPG", y="Price", data=car_data)


# In[46]:


sns.lmplot(x="citympg", y="Price", data=car_data)


# In[47]:


sns.lmplot(x="Popularity", y="Price", data=car_data)


# ## Categorical to Categorical Variables.(X~X)

# In[50]:


pd.crosstab(index=car_data["Make"],columns=car_data["Enginefueltype"])


# In[52]:


pd.crosstab(index=car_data["Make"],columns=car_data["Transmissiontype"])


# In[53]:


pd.crosstab(index=car_data["Make"],columns=car_data["Driven_Wheels"])


# In[54]:


pd.crosstab(index=car_data["Make"],columns=car_data["Marketcategory"])


# In[55]:


pd.crosstab(index=car_data["Make"],columns=car_data["Vehiclesize"])


# In[56]:


pd.crosstab(index=car_data["Make"],columns=car_data["Vehiclestyle"])


# # Multivariate Analysis

# ## Seaborn Pair Plotting

# In[64]:


car_data.head()


# In[65]:


car_data_numeric=car_data.select_dtypes(include=("int64","float64"))


# In[66]:


car_data_numeric.head()


# In[67]:


sns.pairplot(car_data_numeric)


# Inference:
#     This is quite hard to read, and we can rather plot correlations between variables. 
#     Also, a heatmap is pretty useful to visualise multiple correlations in one plot.

# ## Correlation

# In[68]:


car_data_corr=car_data_numeric.corr()


# In[69]:


car_data_corr.to_csv("Cardatacorr.csv")


# In[70]:


plt.figure(figsize=(16,8))
# heatmap
sns.heatmap(car_data_corr, cmap="YlGnBu", annot=True,linewidths=0.1)
plt.show()


# # Selecting important features only

# Inference:
#     
#     Seeing the correlation values of variables with prices
#         Year=0.193556335
#          
#         Numberofdoors=-0.222874548
#          
#         popularity=-0.051834639
#          
#         highwaympg and citympg are highly correlated with each other having 0.921465666 
#         
#         highwaympg and citympg are negatively correlated with price -0.383484515 and -0.393491976
#         coeficients value
#         
#         Hence these variables including model has no such importance ..
# 
# 
# 
# 

# In[73]:


car_data1=car_data[['Make', 'Enginefueltype', 'EngineHP',
       'Enginecylinders', 'Transmissiontype', 'Driven_Wheels',
       'Marketcategory', 'Vehiclesize', 'Vehiclestyle', 'Price']]


# In[74]:


car_data1


# # Data preparation

# In[75]:


car_data1.columns


# In[77]:


X=car_data1[['Make', 'Enginefueltype', 'EngineHP', 'Enginecylinders',
       'Transmissiontype', 'Driven_Wheels', 'Marketcategory', 'Vehiclesize',
       'Vehiclestyle']]


y=car_data1["Price"]


# In[78]:


# creating dummy variables for categorical variables

# subset all categorical variables
car_data1_categorical = X.select_dtypes(include=['object'])
car_data1_categorical.head()


# In[79]:


# convert into dummies
car_data1_dummies = pd.get_dummies(car_data1_categorical, drop_first=True)
car_data1_dummies.head()


# In[80]:


# drop categorical variables 
X = X.drop(list(car_data1_categorical.columns), axis=1)


# In[81]:


# concat dummy variables with X
X = pd.concat([X, car_data1_dummies], axis=1)


# In[82]:


X.shape


# In[83]:


X.columns


# # Scaling the features

# In[84]:


# scaling the features
from sklearn.preprocessing import scale

# storing column names in cols, since column names are (annoyingly) lost after 
# scaling (the df is converted to a numpy array)
cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
X.columns


# # Split into train and test

# In[85]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)


# # Model Building and Evaluation

# In[87]:


# Building the first model with all the features
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
# instantiate
lm = LinearRegression()

# fit
lm.fit(X_train, y_train)


# In[88]:


lm.intercept_


# In[89]:


lm.coef_


# In[90]:


y_train_pred=lm.predict(X_train)
y_test_pred=lm.predict(X_test)


# # Evaluation of Model using various metrics

# In[91]:


from sklearn import metrics


# In[92]:


print("MAE :", metrics.mean_absolute_error(y_train,y_train_pred))
print("MSE : ", metrics.mean_squared_error(y_train,y_train_pred))
print("RMSE : ", np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)) )
print("R^2 : ", metrics.r2_score(y_train,y_train_pred))


# In[93]:


print("MAE :", metrics.mean_absolute_error(y_test,y_test_pred))
print("MSE : ", metrics.mean_squared_error(y_test,y_test_pred))
print("RMSE : ", np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)) )
print("R^2 : ", metrics.r2_score(y_test,y_test_pred))


# Inference:
#     
#     1.Rsquare  for training data is 91.8335%
#       Rsquare  for testing data is 91.2618%

# In[95]:


SS_Residual = sum((y_train-y_train_pred)**2)
SS_Total = sum((y_train-np.mean(y_train))**2)
#r_squared = 1 - (float(SS_Residual))/SS_Total
r_squared=(SS_Total-SS_Residual)/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
print (r_squared, adjusted_r_squared)


# In[97]:


SS_Residual = sum((y_test-y_test_pred)**2)
SS_Total = sum((y_test-np.mean(y_test))**2)
#r_squared = 1 - (float(SS_Residual))/SS_Total
r_squared=(SS_Total-SS_Residual)/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X_train.shape[1]-1)
print (r_squared, adjusted_r_squared)


# Inference:
# 
# 1.Adjusted Rsquare  for training data is 91.61412%
#   Adjusted Rsquare  for testing data is 90.69389%

# # Conclusion

# 1.Rsquare  for training data is 91.8335%
#   Rsquare  for testing data is 91.2618%abs
# 2.1.Adjusted Rsquare for training data is 91.61412% Adjusted Rsquare for testing data is 90.69389%

# # References

# 1.Kaggle
# 2.Purshottam Sir Notes

# In[ ]:




