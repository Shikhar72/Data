#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
os.chdir(r"C:\Users\Admin\Desktop\Training\PGAA001\SVM")


# In[5]:


df = pd.read_csv('voice.csv')
df.head()


# In[6]:


df.isnull().sum()


# In[7]:


print("Total number of labels: {}".format(df.shape[0]))
print("Number of male: {}".format(df[df.label == 'male'].shape[0]))
print("Number of female: {}".format(df[df.label == 'female'].shape[0]))  # Totally balanced data  - (50%-50%)


# In[10]:


X=df.iloc[:, :-1]
X.head()


# In[11]:


p=sns.pairplot(df, hue = 'label') 


# In[12]:


df.head()


# In[8]:


df["Gender"]=np.where(df["label"]=="male",1,0)


# In[9]:


df.drop(columns=["label"], inplace=True)


# In[12]:


# #Converting string value to int type for labels
# from sklearn.preprocessing import LabelEncoder
# y=df.iloc[:,-1]
# # Encode label category
# # male -> 1
# # female -> 0
# gender_encoder = LabelEncoder()
# y = gender_encoder.fit_transform(y)
# y


# In[11]:


y=df["Gender"]
X=df.drop(columns=["Gender"])


# ### Data Standardisation
Standardization refers to shifting the distribution of each attribute to have a mean of zero and 
a standard deviation of one (unit variance). It is useful to standardize attributes for a model. 
Standardization of datasets is a common requirement for many machine learning estimators implemented in 
scikit-learn; they might behave badly if the individual features do not more or less look like standard normally 
distributed data.
# In[19]:


X.head()


# In[16]:


# Scale the data to be between -1 and 1
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)


# In[17]:


# X.head()


# In[15]:


#Splitting dataset into training set and testing set for better generalisation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# In[12]:


#Running SVM with default hyperparameter
from sklearn.svm import SVC # Classifier - Regressor
from sklearn import metrics


# In[13]:


# ?SVC


# In[17]:


svc=SVC() #Default hyperparameters  # kernal - rbf
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
y_pred_TRA=svc.predict(X_train)
print('Accuracy Score:')
print("TEST" ,metrics.accuracy_score(y_test,y_pred))
print("TRAIN", metrics.accuracy_score(y_train,y_pred_TRA))


# In[21]:


svc.get_params(True) #


# In[18]:


svc.predict_proba(X_train)


# In[19]:


#Default Linear kernel
svc=SVC(kernel='linear', probability=True)
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
y_pred_TRA=svc.predict(X_train)
print('Accuracy Score:')
print("TEST" ,metrics.accuracy_score(y_test,y_pred))
print("TRAIN", metrics.accuracy_score(y_train,y_pred_TRA))


# In[24]:


pd.DataFrame(svc.predict_proba(X_train))


# In[24]:


len(svc.support_) # Boundary points ( Rows numbers)


# In[40]:


weight=svc.coef_
intercept=svc.intercept_
print("Wei",weight)
print("Inter", intercept) 


# In[41]:


y_pred=svc.predict(X_train)


# In[42]:


cnf_matrix=metrics.confusion_matrix(y_train, y_pred)
cnf_matrix


# In[43]:


def profile_decile(X,y,trained_model):
    X_1=X.copy()
    y_1=y.copy()
    y_pred1=trained_model.predict(X_1)
    X_1["Prob_Event"]=trained_model.predict_proba(X_1)[:,1]
    X_1["Y_actual"]=y_1
    X_1["Y_pred"]=y_pred1
    X_1["Rank"]=pd.qcut(X_1["Prob_Event"], 10, labels=np.arange(0,10,1))
    X_1["numb"]=10
    X_1["Decile"]=X_1["numb"]-X_1["Rank"].astype("int")
    
    profile=pd.DataFrame(X_1.groupby("Decile")                         .apply(lambda x: pd.Series({
        'min_score'   : x["Prob_Event"].min(),
        'max_score'   : x["Prob_Event"].max(),
        'Event'       : x["Y_actual"].sum(),
        'Non_event'   : x["Y_actual"].count()-x["Y_actual"].sum(),
        'Total'       : x["Y_actual"].count() })))
    return profile
    


# In[44]:


profile_decile(X_train, y_train, svc)


# In[45]:


len(X_train)/10


# In[47]:


profile_train=profile_decile(X_train, y_train, svc)
profile_train.to_csv("profile_train.csv")


# In[46]:


profile_test=profile_decile(X_test, y_test, svc)
profile_test.to_csv("profile_test.csv")


# In[30]:


svc.score(X_train, y_train)


# In[48]:


prob_1=svc.predict_proba(X_train)[:,1]


# In[49]:


y_pred_train_new_cut=np.where(prob_1>0.679,1,0)


# In[50]:


cnf_matrix=metrics.confusion_matrix(y_train, y_pred_train_new_cut)
cnf_matrix


# In[49]:





# In[51]:


#Default RBF kernel
svc=SVC(kernel='rbf')
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
y_pred_train=svc.predict(X_train)
print('Accuracy Score:') 
print(metrics.accuracy_score(y_test,y_pred))
print(metrics.accuracy_score(y_train,y_pred_train))


# In[53]:


#Default Polynomial kernel
svc=SVC(kernel='poly', degree=3, probability=True)
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
y_pred_train=svc.predict(X_train)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))
print(metrics.accuracy_score(y_train,y_pred_train))


# ## Performing K-fold cross validation with different kernels

# ##### CV on Linear kernel

# In[47]:


from sklearn.model_selection import cross_val_score
svc=SVC(kernel='linear')
scores = cross_val_score(svc, X_train, y_train, cv=10, scoring='accuracy') #cv is cross validation
print(scores)


# ##### We can see above how the accuracy score is different everytime.This shows that accuracy score depends upon how the datasets got split

# In[48]:


print(scores.mean()) #In K-fold cross validation we generally take the mean of all the scores


# In[36]:


# #CV on rbf kernel
from sklearn.model_selection import cross_val_score
# svc=SVC(kernel='rbf')
# scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation
# print(scores)
# print(scores.mean())


# In[ ]:


#CV on Polynomial kernel
# from sklearn.model_selection import cross_val_score
# svc=SVC(kernel='poly')
# scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation
# print(scores)


# When K-fold cross validation is done we can see different score in each iteration.
# This happens because when we use train_test_split method,the dataset get split in random manner 
# into testing and training dataset.Thus it depends on how the dataset got split and which samples are 
# training set and which samples are in testing set.
# 
# With K-fold cross validation we can see that the dataset got split into 10 equal parts thus covering all 
# the data into training as well into testing set.This is the reason we got 10 different accuracy score.
# 
# Taking all the values of C and checking out the accuracy score with kernel as linear.
# The C parameter tells the SVM optimization how much you want to avoid misclassifying each training example. 
# For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better 
# job of getting all the training points classified correctly. Conversely, a very small value of C will cause the 
# optimizer to look for a larger-margin separating hyperplane, even if that hyperplane misclassifies more points.
# 
# Thus for a very large values we can cause overfitting of the model and for a very small value of C we can cause 
# underfitting.Thus the value of C must be chosen in such a manner that it generalised the unseen data well

# In[49]:


from sklearn.model_selection import cross_val_score
C_range=[0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9,1]
acc_score=[]
for c in C_range:
    svc = SVC(kernel='linear', C=c)
    scores = cross_val_score(svc, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    acc_score.append(scores.mean())
    print("accuracy for ", c,"is :", scores.mean())  


# In[52]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

C_values=[0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9,1]
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(C_values,acc_score)
plt.xlabel('Value of C for SVC')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# From the above plot we can see that accuracy hasbeen close to 97% for C=1 and C=6 and then 
# it drops around 96.8% and remains constant.

# In[53]:


np.arange(0.1,4,0.1)


# In[54]:


C_range=list(np.arange(0.1,4,0.1))
acc_score=[]
for c in C_range:
    svc = SVC(kernel='linear', C=c)
    scores = cross_val_score(svc, X, y, cv=5, scoring='accuracy')
    acc_score.append(scores.mean())
    print("accuracy for ", c,"is :", scores.mean())


# In[41]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

C_values=list(np.arange(0.1,4,0.1))
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(C_values,acc_score)
plt.xticks(np.arange(0.0,6,0.3))
plt.xlabel('Value of C for SVC ')
plt.ylabel('Cross-Validated Accuracy')


# Taking kernel as rbf and taking different values gamma
# 
# 
# Technically, the gamma parameter is the inverse of the standard deviation of the RBF kernel (Gaussian function), 
# which is used as similarity measure between two points. Intuitively, a small gamma value define a Gaussian function 
# with a large variance. In this case, two points can be considered similar even if are far from each other. 
# In the other hand, a large gamma value means define a Gaussian function with a small variance and in this case, 
# two points are considered similar just if they are close to each other

# In[55]:


gamma_range=[.001,.01, .1, 1, 10,100]
acc_score=[]
for g in gamma_range:
    svc = SVC(kernel='rbf', gamma=g)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
    print("accuracy for ", g,"is :", scores.mean())


# We can see that for gamma=10 and 100 the kernel is performing poorly.We can also see a slight dip in accuracy score when 
# gamma is 1.Let us look into more details for the range 0.0001 to 0.1.

# In[19]:


gamma_range=list(np.linspace(0.1,1,50))
acc_score=[]
for g in gamma_range:
    svc = SVC(kernel='rbf', gamma=g)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
    print("accuracy for ", g,"is :", scores.mean())


# In[20]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
gamma_range=[0.0001,0.001,0.01,0.1]

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(gamma_range,acc_score)
plt.xlabel('Value of gamma for SVC ')
plt.ylabel('Cross-Validated Accuracy')


# In[46]:


gamma_range=[0.01,0.02,0.03,0.04,0.05, 1,10,100,1000]
acc_score=[]
for g in gamma_range:
    svc = SVC(kernel='rbf', gamma=g)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
    print("accuracy for ", g,"is :", scores.mean())


# In[47]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

gamma_range=[0.01,0.02,0.03,0.04,0.05]

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(gamma_range,acc_score)
plt.xlabel('Value of gamma for SVC ')
plt.ylabel('Cross-Validated Accuracy')


# #### Taking polynomial kernel with different degree

# In[56]:


degree=[2,3,4]
acc_score=[]
for d in degree:
    svc = SVC(kernel='poly', degree=d)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
    print("accuracy for ", d,"is :", scores.mean())


# In[49]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
degree=[2,3,4,5,6]
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(degree,acc_score,color='r')
plt.xlabel('degrees for SVC ')
plt.ylabel('Cross-Validated Accuracy')


# Score is high for third degree polynomial and then there is drop in the accuracy score as degree of 
# polynomial increases.Thus increase in polynomial degree results in high complexity of the model and thus causes overfitting.

# In[ ]:





# ### Now performing SVM by taking hyperparameter C=0.1 and kernel as linear

# In[50]:


from sklearn.svm import SVC
svc= SVC(kernel='linear',C=0.1)
svc.fit(X_train,y_train)
y_predict=svc.predict(X_test)
accuracy_score= metrics.accuracy_score(y_test,y_predict)
print(accuracy_score)
y_predict_tra=svc.predict(X_train)
accuracy_score_tra= metrics.accuracy_score(y_train,y_predict_tra)
print(accuracy_score_tra)


# In[24]:


#With K-fold cross validation(where K=10)¶
from sklearn.cross_validation import cross_val_score
svc=SVC(kernel='linear',C=0.1)
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
print(scores)


# In[44]:


#Let us perform Grid search technique to find the best parameter


# In[59]:


from sklearn.svm import SVC
svm_model= SVC(probability=True)


# In[25]:


tuned_parameters = {
'kernel': ['linear','rbf','poly'],

 'degree': [2,3,4] ,'gamma':[0.01,0.02,0.03,0.04,0.05], 'C':(np.arange(0.1,1,0.1)) 
                   }


# In[61]:


from sklearn.model_selection import GridSearchCV
model_svm = GridSearchCV(svm_model, tuned_parameters,cv=10,scoring='accuracy', verbose =2, n_jobs=-1)


# In[62]:


model_svm.fit(X_train, y_train)
print(model_svm.best_score_)


# In[29]:


print(model_svm.best_params_) # finding best parameter


# In[54]:


svc_poly=SVC(C=.9, degree=3,gamma=.05, kernel="poly")


# In[55]:


svc_poly.fit(X_train, y_train)


# In[56]:


svc_poly.score(X_train, y_train)


# In[57]:


svc_poly.score(X_test, y_test)


# In[58]:


y_pred_train= svc_poly.predict(X_train)
print(metrics.accuracy_score(y_pred_train,y_train))


# In[59]:


y_pred= svc_poly.predict(X_test)
print(metrics.accuracy_score(y_pred,y_test))


# In[60]:



svc=SVC(kernel='poly', degree=3, C=0.9, gamma=.05, probability=True)
svc.fit(X_train,y_train)
y_pred_train= svc.predict(X_train)
print(metrics.accuracy_score(y_pred_train,y_train))
y_pred_test= svc.predict(X_test)
print(metrics.accuracy_score(y_pred_test,y_test))


# In[61]:


metrics.confusion_matrix(y_train, y_pred_train)


# In[63]:


print(metrics.classification_report(y_train, y_pred_train))


# In[62]:


probs=svc.predict_proba(X_train)[:,1]
fpr, tpr, threshold=metrics.roc_curve(y_train,probs )


# In[63]:


plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()


# In[42]:


metrics.roc_auc_score(y_train,probs)


# In[53]:


test=pd.read_excel("test.xlsx", sheet_name="Sheet2")
test


# In[54]:


# Scale the data to be between -1 and 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(test)
test = scaler.transform(test)
test


# In[55]:


predicted_test=svc.predict(test)


# In[56]:


predicted_test


# In[57]:


pdff=pd.DataFrame(predicted_test)


# In[58]:


pdff.to_csv("ored.csv")


# In[ ]:




