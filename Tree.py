#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
import os


# In[40]:


#os.chdir("C:\\Users\\Admin\\Desktop\\Training\\DSP25\\DT&RF")
os.chdir("C:\\Users\\Admin\\Desktop\\Training\\PGAA001\\DT")
# C:\Users\Admin\Desktop\Training\DSP31\Python\Treebased


# In[41]:


bank=pd.read_excel('bank.xlsx')
bank.head()


# In[42]:


len(bank.columns) 


# In[43]:


# Check if the data set contains any null values - Nothing found!
bank[bank.isnull().any(axis=1)].count()


# In[44]:


bank.info()


# In[45]:


bank.describe(include='all').T


# In[46]:


# Boxplot for 'age'
g = sns.boxplot(x=bank["age"])


# In[47]:


# Distribution of Age
sns.distplot(bank.age, bins=100)
plt.show()


# In[48]:


# Boxplot for 'duration'
g = sns.boxplot(x=bank["duration"])


# In[49]:


sns.distplot(bank.duration, bins=200)


# ### Convert Categorical Data

# In[50]:


# Make a copy for parsing
bank_data = bank.copy()


# In[51]:


# Explore People who made a deposit Vs Job category
jobs = ['management','blue-collar','technician','admin.','services','retired','self-employed','student',        'unemployed','entrepreneur','housemaid','unknown']
for j in jobs:
    print("{:15} : {:5}". format(j, len(bank_data[(bank_data.deposit == "yes") & (bank_data.job ==j)])))


# In[52]:


# Different types of job categories and their counts
bank_data.job.value_counts()


# In[53]:


# Combine similar jobs into categiroes
bank_data['job'] = bank_data['job'].replace(['management', 'admin.'], 'white-collar')
bank_data['job'] = bank_data['job'].replace(['services','housemaid'], 'pink-collar')
bank_data['job'] = bank_data['job'].replace(['retired', 'student', 'unemployed', 'unknown'], 'other')


# In[54]:


bank_data['job'] = bank_data['job'].replace(['self-employed', 'entrepreneur'], 'Self_Independent')
# New value counts
bank_data.job.value_counts()


# ###### ---------------------------------------------------poutcome-----------------------------------------------

# In[55]:


bank_data.poutcome.value_counts()


# In[56]:


# Combine 'unknown' and 'other' as 'other' isn't really match with either 'success' or 'failure'
bank_data['poutcome'] = bank_data['poutcome'].replace(['other'] , 'unknown')
bank_data.poutcome.value_counts()


# In[ ]:


bank_data['poutcome']


# ##### ---------------------------------------------------contact-----------------------------------------------

# In[57]:


bank_data.contact.value_counts()


# In[58]:


# Drop 'contact', as every participant has been contacted. 
bank_data.drop('contact', axis=1, inplace=True)


# ##### ---------------------------------------------------Default-----------------------------------------------

# In[59]:


# values for "default" : yes/no
bank_data["default"]
bank_data['default_cat'] = bank_data['default'].map( {'yes':1, 'no':0} )
bank_data.drop('default', axis=1,inplace = True)


# ##### ---------------------------------------------------Housing-----------------------------------------------

# In[60]:


bank_data.housing.value_counts()


# In[61]:


# values for "housing" : yes/no
bank_data["housing_cat"]=bank_data['housing'].map({'yes':1, 'no':0})
bank_data.drop('housing', axis=1,inplace = True)


# ##### ---------------------------------------------------loan-----------------------------------------------

# In[62]:


# values for "loan" : yes/no
bank_data["loan_cat"] = bank_data['loan'].map({'yes':1, 'no':0})
bank_data.drop('loan', axis=1, inplace=True)


# ##### ---------------------------------------------------Month & Day-----------------------------------------------

# In[63]:


# day  : last contact day of the month
# month: last contact month of year
# Drop 'month' and 'day' as they don't have any intrinsic meaning
bank_data.drop('month', axis=1, inplace=True)
bank_data.drop('day', axis=1, inplace=True)


# ##### ---------------------------------------------------Deposit-----------------------------------------------

# In[64]:


# values for "deposit" : yes/no
bank_data["deposit_cat"] = bank_data['deposit'].map({'yes':1, 'no':0})
bank_data.drop('deposit', axis=1, inplace=True)


# In[65]:


bank_data.head()


# In[66]:


# pdays: number of days that passed by after the client was last contacted from a
#        previous campaign
#       -1 means client was not previously contacted

print("Customers that have not been contacted before this compaign :", len(bank_data[bank_data.pdays==-1]))
print("Maximum values on padys    :", bank_data['pdays'].max())


# In[67]:


bank_data.columns


# In[68]:


# Map padys=-1 into a large value (10000 is used) to indicate that it is so far in the past that it has no effect
bank_data.loc[bank_data['pdays'] == -1, 'pdays'] = 10000


# In[69]:


# Create a new column:  
bank_data['recent_pdays'] = np.where(bank_data['pdays'], 1/bank_data["pdays"], 
                                     1/bank_data["pdays"])
# Drop 'pdays'
bank_data.drop('pdays', axis=1, inplace = True)


# In[70]:


bank_data.tail()


# ##### -------------Convert to Dummy Values-------------------

# In[71]:


# Convert categorical variables to dummies
bank_with_dummies = pd.get_dummies(data=bank_data, 
                                   columns = ['job', 'marital', 'education', 'poutcome'], \
                                   prefix = ['job', 'marital', 'education', 'poutcome'])
bank_with_dummies.head()


# In[72]:


len(bank_with_dummies.columns)


# In[73]:


bank_data.shape


# In[74]:


bank_with_dummies.shape


# In[75]:


bank_with_dummies.describe()


# ### Bivaraite Analysis

# In[76]:


# Scatterplot showing age and balance
bank_with_dummies.plot(kind='scatter', x='age', y='balance');

# Across all ages, majority of people have savings of less than 20000.


# In[77]:


bank_with_dummies.plot(kind='hist', x='poutcome_success', y='duration');


# In[78]:


#Analysis on people who sign up for a term deposite
# People who sign up to a term deposite
bank_with_dummies[bank_data.deposit_cat == 1].describe()


# In[79]:


# People signed up to a term deposite having a personal loan (loan_cat) and housing loan (housing_cat)
len(bank_with_dummies[(bank_with_dummies.deposit_cat == 1) & (bank_with_dummies.loan_cat) & (bank_with_dummies.housing_cat)])


# In[80]:


# People signed up to a term deposite with a credit default 
len(bank_with_dummies[(bank_with_dummies.deposit_cat == 1) & (bank_with_dummies.default_cat ==1)])


# In[81]:


# Bar chart of job Vs deposite
plt.figure(figsize = (10,6))
sns.barplot(x='job', y = 'deposit_cat', data = bank_data)


# In[82]:


# Y~ All Xs # Highly Related variables -- Cat - Cat - Cross Tab; Freq distribution; Cat- Num - Freq dist, T test , Anova; Num-Num - Scattered plot , Corr  
# Xs~Xs  # Highly Related ??


# In[83]:


# Bar chart of "previous outcome" Vs "call duration"

plt.figure(figsize = (10,6))
sns.barplot(x='poutcome', y = 'duration', data = bank_data)


# In[84]:


# make a copy
bankcl = bank_with_dummies.copy()


# In[85]:


# The Correltion matrix
corr = bankcl.corr()
corr


# In[86]:


# Heatmap
plt.figure(figsize = (10,10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .82})
plt.title('Heatmap of Correlation Matrix')


# In[87]:


# Extract the deposte_cat column (the dependent variable)
corr_deposite = pd.DataFrame(corr['deposit_cat'].drop('deposit_cat'))
corr_deposite.sort_values(by = 'deposit_cat', ascending = False)


# In[89]:


bankcl.to_csv("rf_bank.csv")


# In[ ]:





# In[90]:


# Train-Test split: 20% test data
data_drop_deposite = bankcl.drop('deposit_cat', 1)
label = bankcl.deposit_cat
data_train, data_test, label_train, label_test = train_test_split(data_drop_deposite, label, test_size = 0.2, random_state = 50)


# In[91]:


help(tree.DecisionTreeClassifier)


# In[50]:



# DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, 
#                            min_samples_split=2, min_samples_leaf=1, 
#                            min_weight_fraction_leaf=0.0, max_features=None, 
#                            random_state=None, max_leaf_nodes=None, 
#                            min_impurity_decrease=0.0, 
#                            min_impurity_split=None, 
#                            class_weight=None, 
#                            presort=False)


# In[ ]:



# DecisionTreeClassifier(criterion='gini',  
#                        max_depth=None, 
#                        min_samples_split=2, 
#                        min_samples_leaf=1, 
#                        max_features=None,   
#                        random_state=None,  
#                        class_weight=None)


# In[92]:


# Default
dt2 = tree.DecisionTreeClassifier(random_state=88)


# In[93]:


dt2.fit(data_train, label_train)
dt2_score_train = dt2.score(data_train, label_train)
print("Training score: ",dt2_score_train)
dt2_score_test = dt2.score(data_test, label_test)
print("Testing score: ",dt2_score_test)


# In[68]:


# Decision tree with depth = 10
dt6 = tree.DecisionTreeClassifier(random_state=1, max_depth=10)
dt6.fit(data_train, label_train)
dt6_score_train = dt6.score(data_train, label_train)
print("Training score: ",dt6_score_train)
dt6_score_test = dt6.score(data_test, label_test)
print("Testing score: ",dt6_score_test)


# In[61]:


# Decision tree: To the full depth
dt1 = tree.DecisionTreeClassifier(random_state=1)
dt1.fit(data_train, label_train)
dt1_score_train = dt1.score(data_train, label_train)
print("Training score: ", dt1_score_train)
dt1_score_test = dt1.score(data_test, label_test)
print("Testing score: ", dt1_score_test)


# In[62]:


# Create a feature vector
features = data_drop_deposite.columns.tolist()


# In[63]:


dt1.feature_importances_


# In[66]:


# Fit the decision tree classifier

# fi = dt1.feature_importances_

feature_imp1=pd.DataFrame({"Features":data_train.columns, 
                           "Impo":dt1.feature_importances_}).sort_values("Impo",ascending=False)
feature_imp1.to_excel("FeatureImpodt1.xlsx")
feature_imp1


# In[74]:


# Investigate most important features with depth =6

dt3 = tree.DecisionTreeClassifier(random_state=1, max_depth=6)

# Fit the decision tree classifier
dt3.fit(data_train, label_train)

fi = dt3.feature_importances_

feature_imp3=pd.DataFrame({"Features":data_train.columns, "Impo":dt3.feature_importances_}).sort_values("Impo",ascending=False)
feature_imp3


# In[64]:


feature_imp.to_csv("Feature_imp.csv")


# In[69]:


# According to feature importance results, most importtant feature is the "Duration"
# Let's calculte statistics on Duration
print("Mean duration   : ", data_drop_deposite.duration.mean())
print("Maximun duration: ", data_drop_deposite.duration.max())
print("Minimum duration: ", data_drop_deposite.duration.min())


# In[77]:


from sklearn import tree
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(data_drop_deposite, label, test_size=0.2, random_state=2)

parameters = {'criterion':('gini', 'entropy'), 
              'min_samples_split':[2,3,4,5], 
              'max_depth':[2,4,6,8,9,10,11,12],
              'class_weight':('balanced', None)
 }


tr = tree.DecisionTreeClassifier()
gsearch = GridSearchCV(tr, parameters, cv=10, verbose=1, n_jobs=-1)
gsearch.fit(X_train, y_train)
# model = gsearch.best_estimator_
# model


# In[78]:


gsearch.best_params_


# In[79]:


gsearch.best_score_


# In[80]:


dt4 = tree.DecisionTreeClassifier(random_state=1, max_depth=8,min_samples_split=4, criterion="gini")


# In[81]:


dt4.fit(X_train, y_train)


# In[82]:


dt1_score_train = dt4.score(X_train, y_train)
print("Training score: ", dt1_score_train)
dt1_score_test = dt4.score(X_test, y_test)
print("Testing score: ", dt1_score_test)


# In[83]:


# Fit the decision tree classifier

fi = dt4.feature_importances_

feature_imp4=pd.DataFrame({"Features":X_train.columns, "Impo":dt4.feature_importances_}).sort_values("Impo",ascending=False)

feature_imp4


# In[84]:


feature_imp4.to_csv("feat4.csv")


# In[110]:


data2=data_drop_deposite[['duration','poutcome_success','housing_cat','recent_pdays','age','balance','previous','campaign','loan_cat','job_other','education_tertiary','poutcome_unknown']]


# In[129]:


X_train, X_test, y_train, y_test = train_test_split(data2, label, test_size=0.2, random_state=5)


# In[112]:


dt4.fit(X_train, y_train)


# In[113]:


dt1_score_train = dt4.score(X_train, y_train)
print("Training score: ", dt1_score_train)
dt1_score_test = dt4.score(X_test, y_test)
print("Testing score: ", dt1_score_test)


# In[114]:


fi = dt4.feature_importances_

feature_imp4=pd.DataFrame({"Features":X_train.columns, "Impo":dt4.feature_importances_}).sort_values("Impo",ascending=False)

feature_imp4


# In[ ]:


#Data Clean ; Feature engineering 
#Max depth , Grid search  --> best parameters --> Variable Selection --> Model development


# In[ ]:





# In[92]:


#Accuracy , Recall, precision, F1 score, AUC, ROC, KS , Lift, Selection of p value cut off.


# In[115]:


probs=dt4.predict_proba(X_train)
pd.DataFrame(probs[:,1])


# In[116]:


y_pred=dt4.predict(X_train)
X_train["p_1"]=dt4.predict_proba(X_train)[:,1]
X_train["y_actual"]=y_train
X_train["y_pred"]=y_pred


# In[118]:


X_train["Rank"]=pd.qcut(X_train["p_1"], 10, labels=np.arange(0,10,1))


# In[119]:


X_train["Numb"]=10
X_train["Decile"]=X_train["Numb"]-X_train["Rank"].astype("int")
pd.DataFrame(X_train.groupby("Decile").apply(lambda x:pd.Series({
    "min_score"  :x["p_1"].min(),
    "max_score"  :x["p_1"].max(),
    "Event"      :x["y_actual"].sum(),
    "Non_event"  :x["y_actual"].count()-x["y_actual"].sum(),
    "Total"      :x["y_actual"].count()
})))


# In[ ]:





# In[120]:


profile=pd.DataFrame(X_train.groupby("Decile").apply(lambda x:pd.Series({
    "min_score"  :x["p_1"].min(),
    "max_score"  :x["p_1"].max(),
    "Event"      :x["y_actual"].sum(),
    "Non_event"  :x["y_actual"].count()-x["y_actual"].sum(),
    "Total"      :x["y_actual"].count()
})))


# In[122]:


profile.to_csv("profile.csv")


# In[121]:


y_pred=dt4.predict(X_test)
X_test["p_1"]=dt4.predict_proba(X_test)[:,1]
X_test["y_actual"]=y_test
X_test["y_pred"]=y_pred
X_test["Rank"]=pd.qcut(X_test["p_1"], 10, labels=np.arange(0,10,1))
X_test["Numb"]=10
X_test["Decile"]=X_test["Numb"]-X_test["Rank"].astype("int")
profiletest=pd.DataFrame(X_test.groupby("Decile").apply(lambda x:pd.Series({
    "min_score"  :x["p_1"].min(),
    "max_score"  :x["p_1"].max(),
    "Event"      :x["y_actual"].sum(),
    "Non_event"  :x["y_actual"].count()-x["y_pred"].sum(),
    "Total"      :x["y_actual"].count()
})))


# In[122]:


profiletest.to_csv("profiletest.csv")


# In[123]:


import sklearn.metrics as m 


# In[127]:


X_train.columns


# In[124]:


m.confusion_matrix(X_train['y_actual'],X_train['y_pred'] )


# In[125]:


fpr, tpr, thresholds = metrics.roc_curve(X_train['y_actual'],X_train['y_pred'])
metrics.auc(fpr, tpr)


# In[126]:


print(m.classification_report(X_train['y_actual'],X_train['y_pred']))


# In[ ]:


#Recall,precision, AUC, F1-Score, Accuracy, Lift, GINI index , Decile , Cut off of P,  


# In[127]:



plt.plot(fpr,tpr,label="data 1, auc=")
plt.legend(loc=4)
plt.show()


# In[130]:


dt1_score_train = dt4.score(X_train, y_train)
print("Training score: ", dt1_score_train)
dt1_score_test = dt4.score(X_test, y_test)
print("Testing score: ", dt1_score_test)


# In[131]:


Y_pred_train=dt4.predict(X_train)
Y_pred_test=dt4.predict(X_test)


# In[133]:


confusion_train=m.confusion_matrix(y_train, Y_pred_train)


# In[134]:


confusion_train


# In[137]:


confusion_train[1,0]


# In[141]:


print("Recall Train", confusion_train[1,1]/ (confusion_train[1,1]+confusion_train[1,0]))
print("Precision Train", confusion_train[1,1]/ (confusion_train[1,1]+confusion_train[0,1]))


# In[142]:


confusion_test=m.confusion_matrix(y_test, Y_pred_test)
print("Recall Test", confusion_test[1,1]/ (confusion_test[1,1]+confusion_test[1,0]))
print("Precision Test", confusion_test[1,1]/ (confusion_test[1,1]+confusion_test[0,1]))


# In[148]:


prob_Deposit_train=pd.DataFrame({"Prob_1":dt4.predict_proba(X_train)[:,1]})
prob_Deposit_test=pd.DataFrame({"Prob_1":dt4.predict_proba(X_test)[:,1]})


# In[151]:


y_pred_train_ks=np.where(prob_Deposit_train["Prob_1"]>0.5091,1,0)
y_pred_test_ks=np.where(prob_Deposit_test["Prob_1"]>0.5091,1,0)


# In[150]:


y_pred_train_ks


# In[152]:



confusion_trainks=m.confusion_matrix(y_train, y_pred_train_ks)
print("Recall Train", confusion_trainks[1,1]/ (confusion_trainks[1,1]+confusion_trainks[1,0]))
print("Precision Train", confusion_trainks[1,1]/ (confusion_trainks[1,1]+confusion_trainks[0,1]))


# In[155]:


confusion_testks=m.confusion_matrix(y_test, y_pred_test_ks)
print("Recall Test", confusion_testks[1,1]/ (confusion_testks[1,1]+confusion_testks[1,0]))
print("Precision Test", confusion_testks[1,1]/ (confusion_testks[1,1]+confusion_testks[0,1]))


# In[156]:


from joblib import load 
from joblib import dump


# In[157]:


dump(dt4, "DecisionTree4.joblib")


# In[158]:


Classifier=load('DecisionTree4.joblib')


# In[161]:


new=pd.read_excel("new.xlsx")


# In[162]:


Classifier.predict_proba(new)[:,1]


# In[ ]:




