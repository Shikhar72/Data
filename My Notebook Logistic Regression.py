#!/usr/bin/env python
# coding: utf-8

# # Definition

#  Logistic Regression is used when the dependent variable(target) is categorical.

# For example,
# 
# To predict whether an email is spam (1) or (0)
# 
# Whether the tumor is malignant (1) or not (0)

# Consider a scenario where we need to classify whether an email is spam or not. If we use linear regression for this problem, there is a need for setting up a threshold based on which classification can be done. Say if the actual class is malignant, predicted continuous value 0.4 and the threshold value is 0.5, the data point will be classified as not malignant which can lead to serious consequence in real time.
# 
# From this example, it can be inferred that linear regression is not suitable for classification problem. Linear regression is unbounded, and this brings logistic regression into picture. Their value strictly ranges from 0 to 1.

# # Logistic Function

# The logistic function, also called the sigmoid function 
# 
# It’s an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1, but never exactly at those limits.
# 
# 1 / (1 + e^-value)
# 
# 

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[2]:


img=mpimg.imread("Logistic-Function.png")


# In[3]:


plt.imshow(img)


# # Probability function

# p=exp(y)/(1+exp(y))

# here y is like in linear y=mx+c

# or y=b0+b1x

# p*(1+exp(y))=exp(y)	
# 
# p+p*exp(y)=exp(y)
# 
# exp(y)-p*exp(y)=p
# 
# exp(y)(1-p)=p
# 
# y=log(p/(1-p))	
# 

# # Odds

# p/( 1-p) =	Odds
# 
# y=log(odds)
# 
# 

# Odds Ratio = odds of one / odds of second
# 
# 

# # Confusion Matrix

# A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known

# In[4]:


confusion_matrix_image=mpimg.imread("confusionMatrxiUpdated.jpg")
plt.imshow(confusion_matrix_image)


# Accuracy=How much % of model has correctly classified
# Accuracy=(TP+TN)/total
# 
# Recall=How much % of actually fraud are captured by the data
# 
# Recall=TP/TP+FN
# 
# Recall=Sensitivity=Hit Rate
# 
# Precision=How much % of actual fraud are being captured by the model
# Precision=TP/TP+FP
# Precision=PPV(Postitive Predicted Value )
# 
# F1 score=2*(RECALL*PRECISION)/(RECALL+PRECISION)
# 
# 
# 
# 

# TP=	Values that were actually Positive and predicted as positive			
# FP=	Values that were actually Negative but Predicted as positive			
# FN=	Values that were actually positive but predicted as negative			
# TN=	Value that were actually negative and prdicted as negative 			
# 

# TPR ( True Positive Rate)=	TP/ ( TP+FN)	=		TP/ Actual Positive
# 
# fall out= 	FPR ( False Positive Rate)= 	FP/ ( TN+FP)	=		FP/ actual Negative
# 
# Miss rate =	FNR ( False Negative Rate)=	FN / ( FN + TP )	=		FN/  actual Positive
# 
# Specificity , selectivity =	TNR ( True Negative Rate )=	TN/ ( TN +FP)	=		TN/ Actual Negative
# 					
# Balance Accuracy =	TPR+TNR/2			
# 

# # Loading of Data

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns 


# In[6]:


df=pd.read_sas("bankloan.sas7bdat",encoding="iso-8859-1")


# In[7]:


df


# In[8]:


df.to_csv("Data.csv")


# In[9]:


df.info()


# In[10]:


df.isnull().sum()


# In[11]:


df.columns


# In[12]:


df.default.value_counts(dropna=False)


# In[13]:


df_new_customer=df[df.default.isnull()]


# In[14]:


df_new_customer


# In[19]:


data=df[df.default.notnull()]


# In[20]:


data.to_csv("Custtraindata.csv")


# In[21]:


data


# # Univariate Analysis

# In[22]:


data.info()


# In[23]:


data.columns


# In[24]:


data.nunique()


# All the variables are integer type 

# Variables are not present of object data type.

# In[25]:


data.select_dtypes(include="object").columns


# In[26]:


data.select_dtypes(include="float").columns


# In[27]:


data.describe(percentiles=[.01,.05,.1,.2,.25,.5,.75,.90,.95,.99,1])


# # Bivariate Analysis

# Since all the variables are Numerical

# In[28]:


import matplotlib.pyplot as plt


# In[29]:


plt.scatter(x=data.age,y=data.default)
plt.xlabel("age")
plt.ylabel("default")


# In[30]:


data[["age","default"]].corr()


# as we can see that there is no relation between age and default.

# In[31]:


plt.scatter(x=data.ed,y=data.default)
plt.xlabel("ed")
plt.ylabel("default")


# In[32]:


data[["ed","default"]].corr()


# there is no relation between ed and default

# In[33]:


plt.scatter(x=data.employ,y=data.default)
plt.xlabel("employ")
plt.ylabel("default")


# In[34]:


data[["employ","default"]].corr()


# there is no relation between employ and default.

# In[35]:


import seaborn as sns


# In[36]:


sns.pairplot(data=data)


# In[37]:


data.corr()


# #X~Y analysis
# 
# there is no variable which have 0.5 or more correlation with default variable.

# In[38]:


#X~X analysis


# There is no variables which have more than 0.7 correlation between each other.

# # Outlier Treatment

# In[39]:


a=data.describe(percentiles=[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.40,0.5,0.75, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99,1]).T


# In[40]:


a.to_csv("Outlier-logistic.csv")


# # Training the model

# In[41]:


y=data["default"]


# In[42]:


x=data.drop(columns=["default"])


# In[43]:


x.columns


# In[44]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# In[45]:


from sklearn.linear_model import LogisticRegression


# In[46]:


logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)  # Model train


# # Accuracy

# In[47]:


logisticRegr.score(x_train,y_train)


# In[48]:


logisticRegr.score(x_test,y_test)


# In[49]:


x_train.shape


# In[50]:


logisticRegr.predict(x_train)# by default cut off of p >=0.5  ( if p >=0.5, 1 ,0)


# In[51]:


pd.DataFrame(logisticRegr.predict_proba(x_train)[:,1],columns=["Prob_1"])


# # Prediction and Evaluation of model

# In[52]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[53]:


y_predicted_train=logisticRegr.predict(x_train) # Its predicting 1 or 0


# In[54]:


y_predicted_test=logisticRegr.predict(x_test) # Its predicting 1 or 0


# # Confusion Matrix

# In[55]:


print(classification_report(y_train,y_predicted_train))


# In[56]:


print(classification_report(y_test,y_predicted_test))


# In[57]:


import warnings 
warnings.filterwarnings('ignore')


# In[58]:


#Use score method to get accuracy of model
# score = logisticRegr.score(x_train, y_train)
# print(score)


# In[59]:


cm = metrics.confusion_matrix(y_train, y_predicted_train) # metrics.confusion_matrix(Actual, predicted)
print(cm)


# In[60]:


cmtest = metrics.confusion_matrix(y_test, y_predicted_test) # metrics.confusion_matrix(Actual, predicted)
print(cmtest)


# In[61]:


plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
# all_sample_title = 'Accuracy Score: {0}'.format(score)
# plt.title(all_sample_title, size = 15);


# In[62]:


plt.figure(figsize=(5,4))
sns.heatmap(cmtest, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
# all_sample_title = 'Accuracy Score: {0}'.format(score)
# plt.title(all_sample_title, size = 15);


# # ROC Curve

# It is a plot of the false positive rate (x-axis) versus the true positive rate (y-axis) for a number of different candidate threshold values between 0.0 and 1.0.
# Put another way, it plots the false alarm rate versus the hit rate.

# The true positive rate is calculated as the number of true positives divided by the sum of the number of true positives and the number of false negatives. It describes how good the model is at predicting the positive class when the actual outcome is positive.
# 
# True Positive Rate = True Positives / (True Positives + False Negatives)
# 1
# True Positive Rate = True Positives / (True Positives + False Negatives)
# The true positive rate is also referred to as sensitivity.
# 
# Sensitivity = True Positives / (True Positives + False Negatives)
# 1
# Sensitivity = True Positives / (True Positives + False Negatives)
# The false positive rate is calculated as the number of false positives divided by the sum of the number of false positives and the number of true negatives.
# 
# It is also called the false alarm rate as it summarizes how often a positive class is predicted when the actual outcome is negative.
# 
# False Positive Rate = False Positives / (False Positives + True Negatives)
# 1
# False Positive Rate = False Positives / (False Positives + True Negatives)
# The false positive rate is also referred to as the inverted specificity where specificity is the total number of true negatives divided by the sum of the number of true negatives and false positives.
# 
# Specificity = True Negatives / (True Negatives + False Positives)
# 1
# Specificity = True Negatives / (True Negatives + False Positives)
# Where:
# 
# False Positive Rate = 1 - Specificity
# 1
# False Positive Rate = 1 - Specificity
# The ROC curve is a useful tool for a few reasons:
# 
# The curves of different models can be compared directly in general or for different thresholds.
# The area under the curve (AUC) can be used as a summary of the model skill.
# The shape of the curve contains a lot of information, including what we might care about most for a problem, the expected false positive rate, and the false negative rate.
# 
# To make this clear:
# 
# Smaller values on the x-axis of the plot indicate lower false positives and higher true negatives.
# Larger values on the y-axis of the plot indicate higher true positives and lower false negatives.
# If you are confused, remember, when we predict a binary outcome, it is either a correct prediction (true positive) or not (false positive). There is a tension between these options, the same with true negative and false negative.
# 
# A skilful model will assign a higher probability to a randomly chosen real positive occurrence than a negative occurrence on average. This is what we mean when we say that the model has skill. Generally, skilful models are represented by curves that bow up to the top left of the plot.
# 
# A no-skill classifier is one that cannot discriminate between the classes and would predict a random class or a constant class in all cases. A model with no skill is represented at the point (0.5, 0.5). A model with no skill at each threshold is represented by a diagonal line from the bottom left of the plot to the top right and has an AUC of 0.5.
# 
# A model with perfect skill is represented at a point (0,1). A model with perfect skill is represented by a line that travels from the bottom left of the plot to the top left and then across the top to the top right.
# 
# An operator may plot the ROC curve for the final model and choose a threshold that gives a desirable balance between the false positives and false negatives.

# In[63]:


probs=logisticRegr.predict_proba(x_train)[:,1]


# In[64]:


from sklearn.metrics import roc_curve


# In[65]:


fpr,tpr,thresholds=roc_curve(y_train, probs)


# In[66]:


len(thresholds)


# In[67]:


plt.plot([0, 1], [0, 1], linestyle='--',label="No Skill")
plt.plot(fpr, tpr,label="Logistic")
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()


# # AUC Curve

# Area under curve(roc curve)

# In[68]:


from sklearn.metrics import roc_auc_score

AUC = roc_auc_score(y_train, probs)
AUC


# # Precision Recall Curves

# There are many ways to evaluate the skill of a prediction model.
# 
# An approach in the related field of information retrieval (finding documents based on queries) measures precision and recall.
# 
# These measures are also useful in applied machine learning for evaluating binary classification models.
# 
# Precision is a ratio of the number of true positives divided by the sum of the true positives and false positives. It describes how good a model is at predicting the positive class. Precision is referred to as the positive predictive value.
# 
# Positive Predictive Power = True Positives / (True Positives + False Positives)
# 1
# Positive Predictive Power = True Positives / (True Positives + False Positives)
# or
# 
# Precision = True Positives / (True Positives + False Positives)
# 1
# Precision = True Positives / (True Positives + False Positives)
# Recall is calculated as the ratio of the number of true positives divided by the sum of the true positives and the false negatives. Recall is the same as sensitivity.
# 
# Recall = True Positives / (True Positives + False Negatives)
# 1
# Recall = True Positives / (True Positives + False Negatives)
# or
# 
# Sensitivity = True Positives / (True Positives + False Negatives)
# 1
# Sensitivity = True Positives / (True Positives + False Negatives)
# Recall == Sensitivity
# 1
# Recall == Sensitivity
# Reviewing both precision and recall is useful in cases where there is an imbalance in the observations between the two classes. Specifically, there are many examples of no event (class 0) and only a few examples of an event (class 1).
# 
# The reason for this is that typically the large number of class 0 examples means we are less interested in the skill of the model at predicting class 0 correctly, e.g. high true negatives.
# 
# Key to the calculation of precision and recall is that the calculations do not make use of the true negatives. It is only concerned with the correct prediction of the minority class, class 1.
# 
# A precision-recall curve is a plot of the precision (y-axis) and the recall (x-axis) for different thresholds, much like the ROC curve.
# 
# A no-skill classifier is one that cannot discriminate between the classes and would predict a random class or a constant class in all cases. The no-skill line changes based on the distribution of the positive to negative classes. It is a horizontal line with the value of the ratio of positive cases in the dataset. For a balanced dataset, this is 0.5.
# 
# While the baseline is fixed with ROC, the baseline of [precision-recall curve] is determined by the ratio of positives (P) and negatives (N) as y = P / (P + N). For instance, we have y = 0.5 for a balanced class distribution …
# 
# — The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets, 2015.
# 
# A model with perfect skill is depicted as a point at (1,1). A skilful model is represented by a curve that bows towards (1,1) above the flat line of no skill.
# 
# There are also composite scores that attempt to summarize the precision and recall; two examples include:
# 
# F-Measure or F1 score: that calculates the harmonic mean of the precision and recall (harmonic mean because the precision and recall are rates).
# Area Under Curve: like the AUC, summarizes the integral or an approximation of the area under the precision-recall curve.
# In terms of model selection, F-Measure summarizes model skill for a specific probability threshold (e.g. 0.5), whereas the area under curve summarize the skill of a model across thresholds, like ROC AUC.
# 
# This makes precision-recall and a plot of precision vs. recall and summary measures useful tools for binary classification problems that have an imbalance in the observations for each class.

# In[69]:


from sklearn.metrics import precision_recall_curve


# In[70]:


precision,recall,thresholds=precision_recall_curve(y_train,probs)


# In[71]:


plt.plot([0, 1], [0.52,0.52], linestyle='--',label="No Skill")
plt.plot(precision, recall,label="Logistic")
plt.title("Precision Recall Curve")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend()
plt.show()


# In[72]:


#An AUC score is a measure of the likelihood that the model that produced the
#predictions will rank a randomly chosen positive example above a randomly chosen negative example.
#Specifically, that the probability will be higher for a real event (class=1) than a real non-event 
#(class=0).
#Naive Prediction. A naive prediction under ROC AUC is any constant probability. 
#If the same probability is predicted for every example, there is no discrimination between positive
#and negative cases, therefore the model has no skill (AUC=0.5).

#Insensitivity to Class Imbalance. ROC AUC is a summary on the models ability to correctly 
#discriminate a single example across different thresholds. As such, it is unconcerned with the
#base likelihood of each class.


# In[73]:


#p value, Accuracy, Recall, Precision, ks Values , F1 score, Confusion matrix,ROC, AUC


# # Dumping the Model

# In[74]:


from joblib import dump 
from joblib import load


# In[75]:


dump(logisticRegr, "LogisticReg.joblib")


# In[76]:


log=load("LogisticReg.joblib")


# In[77]:


new=pd.read_excel("new.xlsx")


# In[78]:


log.predict_proba(new)[:,1] # 0.288


# In[79]:


df_new_customer=pd.read_csv("cust_new.csv")


# In[80]:


new_cust=df_new_customer.copy()


# In[81]:


new_cust


# In[82]:


new_cust.drop(columns='default',inplace=True)


# In[83]:


new_cust


# In[90]:


new_cust=new_cust[['age', 'ed', 'employ', 'address', 'income', 'debtinc', 'creddebt',
       'othdebt']]


# In[86]:


new_cust["Prob_default"]=logisticRegr.predict_proba(new_cust)[:,1]


# In[87]:


new_cust["New_Default"]=np.where(new_cust["Prob_default"]>=0.288780649066183,1,0)


# In[88]:


new_cust.New_Default.value_counts()


# In[91]:


new_cust["Predicted_value"]=logisticRegr.predict(new_cust)


# In[92]:


new_cust["Predicted_value"].value_counts()


# # Note

# If we want to increase the Recall,decrease the probability

# If we want to increase the Precision,increase the probability

# In[ ]:




