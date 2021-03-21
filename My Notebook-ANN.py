#!/usr/bin/env python
# coding: utf-8

# # Artificial Neural Network (ANN)

# Artificial Neural Network (ANN) is a deep learning algorithm that emerged and evolved from the idea of Biological Neural Networks of human brains.

# ANN works very similar to the biological neural networks but doesn’t exactly resemble its workings.

# ANN algorithm would accept only numeric and structured data as input.

# To accept unstructured and non-numeric data formats such as Image, Text, and Speech, Convolutional Neural Networks (CNN), and Recursive Neural Networks (RNN) are used respectively.

# # Biological neurons vs Artificial neurons

# ## Structure of Biological neurons and their functions

# Dendrites receive incoming signals.
# 
# Soma (cell body) is responsible for processing the input and carries biochemical information.
# 
# Axon is tubular in structure responsible for the transmission of signals.
# 
# Synapse is present at the end of the axon and is responsible for connecting other neurons.

# ## Structure of Artificial neurons and their functions

# **1.A neural network with a single layer is called a perceptron. A multi-layer perceptron is called Artificial Neural Networks.**
# 
# 
# **2.A Neural network can possess any number of layers. Each layer can have one or more neurons or units. Each of the neurons is interconnected with each and every other neuron. Each layer could have different activation functions as well.**
# 
# **3.ANN consists of two phases Forward propagation and Backpropagation. The forward propagation involves multiplying weights, adding bias, and applying activation function to the inputs and propagating it forward.**
# 
# **4.The backpropagation step is the most important step which usually involves finding optimal parameters for the model by propagating in the backward direction of the Neural network layers. The backpropagation requires optimization function to find the optimal weights for the model.**
# 
# **5.ANN can be applied to both Regression and Classification tasks by changing the activation functions of the output layers accordingly. (Sigmoid activation function for binary classification, Softmax activation function for multi-class classification and Linear activation function for Regression).**
# 

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.image as img


# In[2]:


image=img.imread("ann.png")
plt.figure(figsize=(10,12))
plt.imshow(image)


# # Why Neural Networks?

# 1.Traditional Machine Learning algorithms tend to perform at the same level when the data size increases but ANN outperforms traditional Machine Learning algorithms when the data size is huge
# 
# 2.Feature Learning. The ANN tries to learn hierarchically in an incremental manner layer by layer. Due to this reason, it is not necessary to perform feature engineering explicitly.
# 
# 3.Neural Networks can handle unstructured data like images, text, and speech. When the data contains unstructured data the neural network algorithms such as CNN (Convolutional Neural Networks) and RNN (Recurrent Neural Networks) are used.

# # How ANN works

# The working of ANN can be broken down into two phases,
# 
# Forward Propagation
# 
# Back Propagation

# # Forward Propagation

# 1.Forward propagation involves multiplying feature values with weights, adding bias, and then applying an activation function to each neuron in the neural network.

# 2.Multiplying feature values with weights and adding bias to each neuron is basically applying Linear Regression. If we apply Sigmoid function to it then each neuron is basically performing a Logistic Regression.
# 

# In[3]:


image=img.imread("forward propagation.png")
plt.figure(figsize=(10,12))
plt.imshow(image)


# # Activation functions

# The purpose of an activation function is to introduce non-linearity to the data. Introducing non-linearity helps to identify the underlying patterns which are complex. It is also used to scale the value to a particular interval. For example, the sigmoid activation function scales the value between 0 and 1.

# ## Logistic or Sigmoid function

# Logistic/ Sigmoid function scales the values between 0 and 1.
# 
# It is used in the output layer for Binary classification.
# 
# It may cause a vanishing gradient problem during backpropagation and slows the training time.
# 

# f(x)=1/1+e^-x

# ## Tanh function

# Tanh is the short form for Hyperbolic Tangent. Tanh function scales the values between -1 and 1.

# In[4]:


image=img.imread("1_tAE9A9tDlhnAGXq_-5y2JQ.png")
plt.figure(figsize=(10,12))
plt.imshow(image)


# ## ReLU function

# ReLU (Rectified Linear Unit) outputs the same number if x>0 and outputs 0 if x<0.
# 
# It prevents the vanishing gradient problem but introduces an exploding gradient problem during backpropagation. The exploding gradient problem can be prevented by capping gradients.
# 

# In[5]:


image=img.imread("1_fALttJAfgDC1xxPgHoTpIQ.png")
plt.figure(figsize=(10,12))
plt.imshow(image)


# ## Leaky ReLU function

# Leaky ReLU is very much similar to ReLU but when x<0 it returns (0.01 * x) instead of 0

# If the data is normalized using Z-Score it may contain negative values and ReLU would fail to consider it but leaky ReLU overcomes this problem.

# In[6]:


image=img.imread("1_Uvovl617zz6eJVjQTVi3hQ.png")
plt.figure(figsize=(10,12))
plt.imshow(image)


# ## SoftMax Function

# The softmax function is used as the activation function in the output layer of neural network models that predict a multinomial probability distribution.
# 
# That is, softmax is used as the activation function for multi-class classification problems where class membership is required on more than two class labels.

# In[7]:


image=img.imread("pasted image 0.png")
plt.figure(figsize=(10,12))
plt.imshow(image)


# # Backpropagation

# Backpropagation is done to find the optimal value for parameters for the model by iteratively updating parameters by partially differentiating gradients of the loss function with respect to the parameters.
# 
# 
# An optimization function is applied to perform backpropagation. The objective of an optimization function is to find the optimal value for parameters.

# The optimization functions available are:
#     
#     Gradient Descent
#     
#     Adam optimizer
#     
#     Gradient Descent with momentum
#     
#     RMS Prop (Root Mean Square Prop)

# The Chain rule of Calculus plays an important role in backpropagation. The formula below denotes partial differentiation of Loss (L) with respect to Weights/ parameters (w).

# A small change in weights ‘w’ influences the change in the value ‘z’ (∂𝑧/∂𝑤). A small change in the value ‘z’ influences the change in the activation ‘a’ (∂a/∂z). A small change in the activation ‘a’ influences the change in the Loss function ‘L’ (∂L/∂a).

# In[9]:


image=img.imread("lossfn.PNG")
plt.figure(figsize=(10,12))
plt.imshow(image)


# # Terminologies:

# # Metrics

# 
# 
# A metric is used to gauge the performance of the model.
# 
# Metric functions are similar to cost functions, except that the results from evaluating a metric are not used when training the model. Note that you may use any cost function as a metric.
# We have used Mean Squared Logarithmic Error as a metric and cost function.
# 

# In[10]:


image=img.imread("1_bgzceS9jybK6YzuRrGTbPw.png")
plt.figure(figsize=(10,12))
plt.imshow(image)


# # Epoch

# A single pass through the training data is called an epoch. The training data is fed to the model in mini-batches and when all the mini-batches of the training data are fed to the model that constitutes an epoch.

# # Hyperparameters

# Hyperparameters are the tunable parameters that are not produced by a model which means the users must provide a value for these parameters. The values of hyperparameters that we provide affect the training process so hyperparameter optimization comes to the rescue.
# The Hyperparameters used in this ANN model are,
# 
# Number of layers
# 
# Number of units/ neurons in a layer
# 
# Activation function
# 
# Initialization of weights
# 
# Loss function
# 
# Metric
# 
# Optimizer
# 
# Number of epochs
# 
# 

# # Coding

# ##  Business Problem

# 
# Our basic aim is to predict customer churn for a certain bank i.e. which customer is going to leave this bank service. Dataset is small(for learning purpose) and contains 10000 rows with 14 columns.

# ## Churn classification using Keras

# In[1]:


import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt


# In[3]:


dataset = pd.read_csv('Churn_Modelling (1).csv')


# In[5]:


#import pandas_profiling
#pandas_profiling.ProfileReport(dataset)


# In[6]:


dataset.head()


# In[7]:


dataset.shape


# In[8]:


dataset.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)


# In[9]:


dataset.shape


# In[10]:


dataset.head()


# In[11]:


dataset_new = pd.get_dummies(dataset, ['Geography', 'Gender'], drop_first=True)
dataset_new.head()


# In[12]:


dataset_new.shape


# In[13]:


dataset_new.columns


# In[14]:


X = dataset_new.drop(columns=["Exited"])


# In[15]:


y = dataset_new['Exited']


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state =123)


# In[18]:


#Scaling variables - Helps to converge quickly
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)


# In[19]:


train_X.shape


# In[20]:


test_X.shape


# In[21]:


type(train_X)


# In[22]:


from keras.models import Sequential
from keras import activations, initializers, regularizers, constraints
from keras.layers import Dense, Activation


# # Calling the Model

# In[23]:


model = Sequential()


# # Adding the layer

# In[24]:


#NEtwork = [11, 6, 4,1]


# In[25]:


model.add(Dense(units=6,activation='relu',kernel_initializer="uniform",input_dim=11))
model.add(Dense(units=4,activation="relu",kernel_initializer="uniform"))
model.add(Dense(units=1,activation="sigmoid",kernel_initializer="uniform"))


# # Compiling and fitting

# In[26]:


model.compile(optimizer="adam",loss='binary_crossentropy',metrics=["accuracy"])


# In[27]:


model.fit(train_X,train_y,batch_size=10,epochs=10)


# # Prediction

# In[29]:


y_pred=model.predict(test_X)


# In[30]:


y_pred           ##it will give only the probability value not the class value


# In[31]:


import sklearn.metrics as metrics


# In[32]:


metrics.roc_auc_score(test_y, y_pred)


# In[33]:


y_pred1 = np.where(y_pred>0.4,1,0)
y_pred1


# Using np.where to predict the classes.

# In[34]:


from sklearn.metrics import confusion_matrix


# In[35]:


cm=confusion_matrix(test_y,y_pred1)


# In[36]:


cm


# In[37]:


from sklearn.metrics import classification_report


# In[38]:


print(classification_report(test_y,y_pred1))


# In[39]:


model.save('model.h5')


# In[40]:


from keras.models import load_model


# In[41]:


model1 = load_model('model.h5')


# In[43]:


model1.summary()


# In[44]:


from keras.utils.vis_utils import plot_model
plot_model(model1,to_file='model_plot.png',show_layer_names=True,show_shapes=True)


# In[ ]:




