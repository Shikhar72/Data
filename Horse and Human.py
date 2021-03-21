#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys


# In[2]:


sys.version


# # Importing different Libraries

# In[3]:


import pandas as pd
import keras
from keras.models import  Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[5]:


import warnings
warnings.filterwarnings("ignore")


# # Create Architecture/Framework

# In[31]:


model1=Sequential()


# In[32]:


model1.add(Conv2D(filters=40,kernel_size=3,activation="relu",input_shape=(64,64,3)))
model1.add(MaxPooling2D(pool_size=2))


# In[33]:


model1.add(Conv2D(filters=32,kernel_size=3,activation="relu"))
model1.add(MaxPooling2D(pool_size=2))


# In[34]:


model1.add(Conv2D(filters=40,kernel_size=3,activation="relu"))
model1.add(MaxPooling2D(pool_size=2))


# In[35]:


model1.add(Conv2D(filters=45,kernel_size=3,activation="relu"))
model1.add(MaxPooling2D(pool_size=2))


# In[36]:


model1.add(Flatten())


# In[37]:


model1.summary()


# # Creating ANN Architecture over CNN Architecture

# In[38]:


model1.add(Dense(units=180,activation='relu',kernel_initializer='uniform'))
model1.add(Dense(units=180,activation='relu',kernel_initializer='uniform'))
model1.add(Dense(units=180,activation='relu',kernel_initializer='uniform'))
model1.add(Dense(units=150,activation='relu',kernel_initializer='uniform'))
model1.add(Dense(units=150,activation='relu',kernel_initializer='uniform'))
model1.add(Dense(units=150,activation='relu',kernel_initializer='uniform'))
model1.add(Dense(units=150,activation='relu',kernel_initializer='uniform'))
model1.add(Dense(units=120,activation='relu',kernel_initializer='uniform'))

model1.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))


# In[39]:


model1.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])


# # Using Image Data Generator for train and test data splitting

# In[40]:


from keras.preprocessing.image import ImageDataGenerator


# In[41]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


# In[42]:


test_datagen = ImageDataGenerator(rescale=1./255)


# In[43]:


train_set = train_datagen.flow_from_directory (r"C:\Users\alokk\OneDrive\Desktop\Project\Train",    target_size=(64, 64),    batch_size=32, class_mode='binary')


# In[44]:


test_set = test_datagen.flow_from_directory(r"C:\Users\alokk\OneDrive\Desktop\Project\Test",        target_size=(64, 64),        batch_size=32,        class_mode='binary')


# In[45]:


history=model1.fit_generator(
        train_set,
        steps_per_epoch=1027,
        epochs=5,
        validation_data=test_set,
        validation_steps=256)


# In[46]:


x1=history.history["accuracy"]


# In[47]:


x2=history.history["loss"]


# In[48]:


import matplotlib.pyplot as plt


# In[49]:


plt.plot(range(1,6),x1)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.show()


# In[50]:


plt.plot(range(1,6),x2)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


# In[53]:


plt.plot(range(1,6),x1,label="Accuracy")
plt.plot(range(1,6),x2,label="Loss")
plt.xlabel("epochs")
plt.ylabel("metrics")
plt.xlim(1,6)
plt.legend()
plt.show()


# # Loading  the image

# In[ ]:


import numpy


# In[ ]:


from keras.preprocessing import images


# In[ ]:




