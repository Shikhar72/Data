#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
test_list = [100,200,300]
pd.Series(data=test_list)


# In[2]:


test_list1=[[1,2,3],["a","b","c"]]
pd.Series(data=test_list1)


# In[3]:


test_list2=[[1,2,3],["a","b","c","d"]]
pd.Series(data=test_list2)


# In[4]:


test_list2=[[1,2,3],["a","b","c","d"]]
pd.Series(data=test_list2,index=("a","b"))


# In[5]:


mat=np.random.randint(10,100,(10,4))
mat


# In[6]:


print("Section A has an average of :",mat[:,0].mean())
print("Section B has an average of :",mat[:,1].mean())
print("Section C has an average of :",mat[:,2].mean())
print("Section D has an average of :",mat[:,3].mean())


# In[7]:


df=pd.DataFrame(data=mat,columns=["SectionA","SectionB","SectionC","SectionD"])
df


# In[8]:


df.mean()


# In[9]:


#axis=0>column wise>default
df.mean(axis=0)


# In[10]:


#axis=0>row wise
df.mean(axis=1)


# # Pandas Series

# In[12]:


pd.Series(data=test_list,columns="century")


# pandas series have no attribute columns

# In[13]:


test_dictionary={"a":100,"b":200,"c":300}
pd.Series(data=test_dictionary)


# # Pandas DataFrame

# In[14]:


t3_data=[["alok",24],["ankit",25],["sakshi",25]]
df=pd.DataFrame(data=t3_data,columns=["name","age"])
df


# In[15]:


t3_data=[["alok",24],["ankit",25],["sakshi",25]]
df=pd.DataFrame(data=t3_data,columns=["name","age"],index="name")
df


# In[16]:


t3_data=[["alok",24],["ankit",25],["sakshi",25]]
df=pd.DataFrame(data=t3_data,columns=["name","age"],index=["a","b","c"])
df


# Since the columns were not been already defined.
# Hence index is not being set.

# In[17]:


t3_data=[["alok",24,"student"],["ankit",25,"student"],["sakshi",25,"administrator"]]
df=pd.DataFrame(data=t3_data,columns=["name","age","role"])
df


# In[18]:


t3_data=np.array([[["alok",24,"student"],["ankit",25,"student"]],[["sakshi",25,"administrator"],["abhi",24,"nonmember"]]])
print(t3_data.ndim)
df=pd.DataFrame(data=t3_data,columns=["name","details"])
df


# We have to give only 2d input 3d input is not allowed here

# In[19]:


t3_dictionary={"name":["alok","ankit","sakshi"],"age":[24,25,25]}
df=pd.DataFrame(data=t3_dictionary)
df


# # Reading Files

# read_csv,read_excel,read_txt,read_sas,read_html,read_sql

# df = pd.read_csv(filepath)

# In[20]:


df=pd.read_csv("Titanic_train.csv")
df
df.columns


# In[21]:


#Read the csv file with index
dat=pd.read_csv("Titanic_train.csv",index_col=3)
dat
dat.columns


# In[22]:


#or
dat=pd.read_csv("Titanic_train.csv",index_col="Name")
dat
dat.columns


# Analyse the columns name and see

# In[23]:


data=pd.read_csv('https://s3-eu-west-1.amazonaws.com/shanebucket/downloads/uk-500.csv')
data


# In[ ]:


data.set_index("last_name",inplace=True)
data


# # Summary of the dataset

# In[ ]:


df=pd.read_csv("Titanic_train.csv")


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.max()


# In[ ]:


df.min()


# In[ ]:


df.std()


# In[ ]:


df.count()


# In[ ]:





# # Unique Values for each column

# In[ ]:


df.nunique()


# In[ ]:


df["Pclass"].unique()


# The df.unique() command allows us to better understand what does each column mean. Looking at the Survived and Sex columns, they only have 2 unique values. It usually means that they are categorical columns, in this case, it should be True or False for Survived and Male or Female for Sex.
# We can also observe other categorical columns like Embarked, Pclass and more. We can’t really tell what does Pclass stand for, let’s explore more.

# # Selection of Data

# In[ ]:


df["Pclass"]


# We observe that the 3 unique values are literally 1,2 and 3 which stands for 1st class, 2nd class and 3rd class.

# # Selecting different columns at once

# In[ ]:


df[["Pclass","Sex"]]


# In[ ]:


df["Pclass"][8]


# This command is extremely useful for including/excluding columns you need/don’t need.

# Since we don’t need the name, passenger_id and ticket because they play no role at all in deciding if the passenger survives or not, we are excluding these columns in the dataset and it can be done in 2 ways.

# # Droping of Particular columns

# In[ ]:


df= df[["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Cabin","Embarked"]]


# In[ ]:


df


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


# df.drop(["Name","Ticket","PassengerId"],axis=1,inplace=True)


# In[ ]:


df=pd.read_csv("Titanic_train.csv")
df


# In[ ]:


#axis=1>>here columns 
#axis=0>>here index
df.drop(["Name","Ticket","PassengerId"],axis=1,inplace=True)
df


# The inplace=True parameter tells Pandas to auto assign what you intend to do to the original variable itself, in this case it is df.

# # iloc and loc function

# In[ ]:


df.iloc[500:511]


# In[ ]:


df.iloc[[1,3,4],[1,4,5]]


# In[ ]:


df.iloc[1:5,5:8]


# In[ ]:


df.loc[550:558,["Survived","Fare"]]


# It works just like a python list, where the first number(500) of the interval is inclusive and the last number(511) of the interval is exclusive.

# # Selecting a column and its one Type

# In[ ]:


a=df[df["Sex"]=="male"]
b=df[df["Sex"]=="female"]
print(a)
print(b)


# In[ ]:


type(a)


# # counting of values

# In[24]:


df["Sex"].value_counts()


# it automatically predicts how many unique values are there,here Sex have two unique values

# In[25]:


a=df[df["Sex"]=="male"]
a


# The command df[‘Sex’] == ‘male’ will return a Boolean for each row. Nesting a df[] over it will return the whole dataset for male passengers.
# That’s how Pandas work.

# In[26]:


df[["Pclass","Sex"]][df["Sex"]=='male'].iloc[500:511]


# With this command, we are displaying only 500th to 510th row of the Pclass and Sex Column for Male Passengers

# In[27]:


df[["Pclass","Sex"]][df["Sex"]=="female"].iloc[110:570]


# # Aggregation Functions

# In[28]:


df.describe()


# In[29]:


df.max()


# In[30]:


df['Survived'].max()


# In[31]:


df.std()


# In[32]:


df.mean()


# In[33]:


df.iloc[100:151].max()


# In[34]:


df["Age"].unique()


# In[35]:


df['Age'].value_counts()


# In[36]:


df.count()


# In[37]:


df.corr()


# It looks like top 3 numerical columns that are contributing to the survivability of the passengers are Fare, Pclass and Parch because they hold the highest absolute values against the survived column.

# We can also check the distribution of each column. In a table format, it is easier to interpret distributions for categorical columns.

# # Data Cleaning

# Your dataset can often include dirty data like:
#     
# null values
# empty values
# incorrect timestamp
# many many more

# In[38]:


df.info()


# In[39]:


df.isnull().sum()


# We can see that there are null values in the Age, Cabin and Embarked columns. You can see what the rows look like when the values are null in those specific columns.

# In[40]:


df["Age"].isnull().sum()


# In[41]:


df[df["Age"].isnull()]


# For simplicity’s sake, we will remove all rows with null values in them.

# While reading the file we can check the  null values itself.

# In[42]:


df=pd.read_csv("Titanic_train.csv",na_filter=False)
df


# In[43]:


df=pd.read_csv("Titanic_train.csv",na_values=df["Age"].mean(),na_filter=False)


# here we have directly filter the null values and tried to fill all the null values with the mean of age,
# hence  we got an error that we can not replace the string null value with the mean of mean of integer value

# In[44]:


import warnings
warnings.filterwarnings("ignore")


# In[45]:


df.dropna()


# In[46]:


#df.dropna(axis=1, inplace=True)


# In[47]:


df=pd.read_csv("Titanic_train.csv")


# In[48]:


df["Age"].fillna(df["Age"].mean())


# # Pivot Table

# In[49]:


df


# In[50]:


df.pivot_table(index=["Sex","Pclass"],columns="Survived",values="Age",aggfunc=np.mean)


# In[51]:


df.pivot_table(index=["Sex","Pclass"],columns="Survived",values="Age",aggfunc=np.mean,fill_value=df["Age"].mean())


# In[52]:


df.pivot_table(index=["Sex","Pclass"],columns=["Age","Embarked","Fare"],values="Survived",aggfunc=np.sum)


# In[53]:


df.pivot_table(index=["Sex"],aggfunc={"Age":np.mean,"Fare":np.sum})


# In[54]:


df.pivot_table(columns="Pclass",values="Survived")


# # GroupBy

# In[55]:


df


# In[56]:


df.groupby("Pclass")


# In[57]:


df.groupby("Pclass").mean()


# In[58]:


df.groupby(['Pclass','Age']).mean()


# In[59]:


df.groupby(['Pclass','Sex']).mean()


# In[60]:


df.groupby(['Sex','Pclass']).mean()


# One observation I can make is that the Average Fare for Females in 1st Class is the highest among all the passengers, and they have the highest survivability as well. On the flip side, the Average Fare for Males in
# 3rd Class is the lowest and has the lowest survivability.

# In[61]:


t=pd.DataFrame(
    {
        'Price':[10,11,12,14,15,16,18,10,11,12],
        'Accessories':['table','chair','chair','tablet','computer','table','chair','chair','tablet','computer']
    }
)
t


# In[62]:


t.shape


# In[63]:


t.groupby("Accessories")["Price"].mean()


# In[64]:


type(t)


# # Concatenation and Merging

# In[65]:


first_5=df.head()
last_5=df.tail()
combined=pd.concat([last_5,first_5])
combined


# In[66]:


combined1=pd.concat([df,t])
combined1


# In[67]:


combined1=pd.concat([df,t],axis=1)
combined1


# just analyse the  the number of rows and columns for understanding the axis

# In[68]:


a=np.arange(1,5)
b=np.arange(6,11)
c=np.concatenate((a,b))
c


# # Merging

# Now what if we want to add columns by joining.
# We merge.

# In[69]:


df=pd.read_csv("Titanic_train.csv")


# In[70]:


data = [["Braund, Mr. Owen Harris", 80,  177.0], ['Heikkinen, Miss. Laina', 78, 180.0], ['Montvila, Rev. Juozas', 87, 165.0]] 
df2 = pd.DataFrame(data, columns = ['Name', 'weight', 'height'])


# In[71]:


df2


# In[72]:


df3=pd.merge(df,df2,how="right",on="Name",left_on=None, right_on=None, left_index=False, right_index=False)
df3


# In[73]:


df4=pd.merge(df,df2,how="left",on="Name")
df4


# # Changing the Data Types

# In[82]:



df2["height"].astype(int)


# In[78]:


a=14.5
print(int(a))


# # Apply Function

# Step 1: Define a function
# 
# Assume we want to rename the Pclass to their actual names.
# 
# 1 = 1st Class
# 
# 2 = 2nd Class
# 
# 3 = 3rd Class
# 
# We first define a function.

# In[75]:


def change_name(x):
    if x == 1:
        x = '1st Class'
    if x == 2:
        x = '2nd Class'
    if x == 3:
        x = '3rd Class'
    return x
 


# In[76]:


df=pd.read_csv("C:/Users/Dell/Downloads/titanic_train.csv")


# In[ ]:


df


# Step 2: Apply the function
# We can then apply the function to each of the values of Pclass column.
# The process will pass every record of the Pclass column through the function.

# In[ ]:


df['Pclass']=df["Pclass"].apply(lambda x: change_name(x))
df


# In[ ]:


df["Pclass"]=df["Pclass"].map(change_name)
df


# # Custom Columns

# Sometimes, we want to create custom columns to allow us to better interpret our data. For example, let’s create a column with the BMI of the passengers.

# The formula for BMI is:
# 
# weight(kg) / height²(m)

# In[ ]:


def convert_to_bmi(x):
    x = (x/100)**2
    return x
df4["height"] = df4["height"].apply(lambda x: convert_to_bmi(x))


# In[ ]:


df4


# In[ ]:


df4['bmi']=df4['weight']/df4['height']
df4


# # Rename

# In[ ]:


df4.rename(columns={'bmi':'Body_Mass_Index','PassengerId':'PassengerNo'}, inplace = True)


# In[ ]:


df4


# # Sorting

# In[ ]:


df4=df4.sort_values("Body_Mass_Index")
df4


# # Exporting The file

# In[ ]:


df4.to_csv("C:/Users/Dell/Downloads/df4titanic.csv")

