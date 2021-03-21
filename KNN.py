#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install pywhatkit


# In[1]:


import pywhatkit
pywhatkit.sendwhatmsg("+917828408203", "Hi Veer ", 10,24)


# In[3]:


pywhatkit.info("Youtube", lines=5)


# In[4]:


import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import  metrics
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')


# In[5]:


os.chdir(r"C:\Users\Admin\Desktop\Training\PGAA001\KNN")


# In[6]:


diabetes_data = pd.read_csv('D.csv')


# In[7]:


diabetes_data.head()


# In[8]:


diabetes_data.info()


# In[11]:


diabetes_data.describe().T


# In[9]:


diabetes_data.isnull().sum()


# In[11]:


diabetes_data.columns.to_list()


# In[13]:


diabetes_data_copy = diabetes_data.copy()
diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = 
diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

## showing the count of Nans
print(diabetes_data_copy.isnull().sum())


# In[15]:


p = diabetes_data.hist(figsize = (20,20))


# In[14]:


diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)

diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)

diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)

diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)

diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)


# In[17]:


p = diabetes_data_copy.hist(figsize = (20,20))


# In[18]:


diabetes_data.dtypes


# In[19]:


#import missingno as msno
#p=msno.bar(diabetes_data)


# In[15]:


diabetes_data_copy["Outcome"].value_counts()


# In[16]:


268/(500+268)  # Close to balance - 


# In[17]:


## checking the balance of the data by plotting the count of outcomes by their value
color_wheel = {1: "#0392cf", 
               2: "#7bc043"}
colors = diabetes_data["Outcome"].map(lambda x: color_wheel.get(x + 1))
print(diabetes_data.Outcome.value_counts())
p=diabetes_data.Outcome.value_counts().plot(kind="bar")


# In[25]:


# from pandas.tools.plotting import scatter_matrix
# p=scatter_matrix(diabetes_data,figsize=(25, 25))


# In[18]:


plt.scatter(diabetes_data_copy["Age"],diabetes_data_copy["Glucose"])
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.show()


# In[28]:


sns.pairplot(diabetes_data_copy)


# In[29]:


corr_mat=diabetes_data_copy.corr()
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
sns.heatmap(corr_mat, annot=True,cmap ='RdYlGn')  


# In[19]:


# plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
# p=sns.heatmap(diabetes_data_copy.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap


# In[20]:


d=diabetes_data_copy.corr()
d.to_excel("corr.xlsx" )


# In[23]:


from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
sc_X = StandardScaler()
sc_X.fit_transform(diabetes_data_copy)


# In[24]:


X=diabetes_data_copy.drop(["Outcome"],axis = 1)


# In[25]:


y=diabetes_data_copy["Outcome"]


# In[33]:


from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
sc_X = StandardScaler()

X1=sc_X.fit_transform(X)

X=pd.DataFrame(X1, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])


# In[38]:


X


# In[35]:


X.head()


# In[76]:


#importing train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)


# In[40]:


X_train.shape


# In[41]:


np.sqrt(X_train.shape[0])


# In[44]:


# help(KNeighborsClassifier)


# In[47]:



knn = KNeighborsClassifier(23)
knn.fit(X_train,y_train)
print("Score on Train data",knn.score(X_train,y_train))
print("Score on Test data",knn.score(X_test,y_test))


# In[48]:


test_scores = []
train_scores = []
for i in range(1,50):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)  
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))


# In[50]:


test_scores


# In[58]:


plt.figure(figsize=(12,5))
sns.lineplot(range(1,50),train_scores,marker='*',label='Train Score')
sns.lineplot(range(1,50),test_scores,marker='o',label='Test Score')


# In[55]:


df=pd.DataFrame({"K":range(1,50), "train":train_scores, "Test":test_scores})
df["Diff"]=df["train"]-df["Test"]
df.to_csv("fg.csv")


# In[61]:



#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(11)

knn.fit(X_train,y_train)
print("Score on Train data",knn.score(X_train,y_train))
print("Score on Test data",knn.score(X_test,y_test))
print( knn.score(X_train,y_train)-knn.score(X_test,y_test))


# In[62]:


#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(23)
knn.fit(X_train,y_train)
print("Score on Train data",knn.score(X_train,y_train))
print("Score on Test data",knn.score(X_test,y_test))
print( knn.score(X_train,y_train)-knn.score(X_test,y_test))


# In[63]:


#import confusion_matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[64]:


#import confusion_matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_train)
confusion_matrix(y_train,y_pred)
pd.crosstab(y_train,y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[65]:


y_pred = knn.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[66]:


#import classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[67]:


from sklearn.metrics import roc_curve
y_pred_proba = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


# In[68]:


plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=11) ROC curve')
plt.show()


# In[69]:


p=knn.predict_proba(X_test)


# In[70]:


y_pred=knn.predict(X_train)
X_train["p_1"]=knn.predict_proba(X_train)[:,1]
X_train["y_actual"]=y_train
X_train["y_pred"]=y_pred


# In[71]:


#Area under ROC curve
from sklearn.metrics import roc_auc_score
y_pred_proba=p[:,1]
print(roc_auc_score(y_test,y_pred_proba) )
# print(.81-.76)


# In[56]:


fg=pd.DataFrame(p)


# In[57]:


fg.to_csv("prob.csv")
y_test.to_csv("Y_test.csv")
pd.DataFrame(y_pred).to_csv("Y_pred.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[41]:


help(knn)


# In[81]:


#import GridSearchCV
from sklearn.model_selection import GridSearchCV
#In case of classifier like knn the parameter to be tuned is n_neighbors
param_grid = {'n_neighbors':np.arange(3,5), "p":[1,2,3]}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5, verbose=1)
knn_cv.fit(X_train,y_train)
print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_)) 


# In[82]:


#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=3,  p=1)
knn.fit(X_train,y_train)
print("Score on Train data",knn.score(X_train,y_train))
print("Score on Test data",knn.score(X_test,y_test))


# In[84]:


knn_cv.cv_results_


# In[92]:


print(knn_cv.cv_results_["split4_test_score"])


# In[ ]:




