#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network

# ### Importing the libraries

# In[1]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


tf.__version__


# ## Part 1 - Data Preprocessing

# ### Preprocessing the Training set

# In[3]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


# ### Preprocessing the Test set

# In[4]:


# No image augmentation is applied, only feature scalling was applied
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


# ## Part 2 - Building the CNN

# ### Initialising the CNN

# In[5]:


cnn = tf.keras.models.Sequential()


# ### Step 1 - Convolution

# In[6]:


cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size = 3, activation='relu', input_shape=[64, 64, 3]))


# ### Step 2 - Pooling

# In[7]:


cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))


# ### Adding a second convolutional layer

# In[8]:


cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size = 3, activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))


# ### Step 3 - Flattening

# In[9]:


cnn.add(tf.keras.layers.Flatten())


# ### Step 4 - Full Connection

# In[10]:


cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))


# ### Step 5 - Output Layer

# In[23]:


cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
""" Note: If we have more than 2 catagories of output like: cat, do, rat then activation of
          output layer is 'softmax'  and units(output_dim) = 3,i.e.same as number of outputs.
          In above case since output is binar : cat or dog so it is sigmoid.
""" 


# ## Part 3 - Training the CNN

# ### Compiling the CNN

# In[24]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
"""Note: If you have more than 2 categories of output like: cat, dog, rat (more than 2 output) then
   loss = 'catagorical_crossentropy'. In above case we have binary oytcome: cat or dog.
"""


# ### Training the CNN on the Training set and evaluating it on the Test set

# In[13]:


cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


# ## Part 4 - Making a single prediction

# In[21]:


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_4.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0) # extra dimension (axis = 0 means first dimension for batch)corresponds to batch which has eaxcly right format required by our prediction method
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1: ## [0]-> accesing a batch, [0]->single element of this batch
    prediction = 'dog'
else:
    prediction = 'cat'


# In[22]:


print(prediction)


# In[16]:


#pip install pillow


# In[ ]:




