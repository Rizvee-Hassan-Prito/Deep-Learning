# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 00:09:51 2022

@author: User
"""

import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Import the Conv2D and Flatten layers and instantiate model
from tensorflow.keras.layers import Conv2D,Flatten
model = Sequential()

# Add a convolutional layer of 32 filters of size 3x3
model.add(Conv2D(filters = 32, kernel_size = 3, input_shape = (28, 28, 1), activation = 'relu'))

# Add a convolutional layer of 16 filters of size 3x3
model.add(Conv2D(filters = 16, kernel_size = 3, activation = 'relu'))

# Flatten the previous layer output
model.add(Flatten())

# Add as many outputs as classes with softmax activation
model.add(Dense(10, activation = 'softmax'))

model.summary()

#%%

import tensorflow as tf
import matplotlib.pyplot as plt 

mnist= tf.keras.datasets.mnist

(x_train,y_train), (X_test,y_test)= mnist.load_data()

x_train=tf.keras.utils.normalize(x_train,axis=1)
X_test=tf.keras.utils.normalize(X_test,axis=1)

#%%
from tensorflow.keras.models import Model

# Obtain a reference to the outputs of the first layer
first_layer_output = model.layers[0].output

# Build a model using the model's input and the first layer output
first_layer_model = Model(inputs = model.layers[0].input, outputs = first_layer_output)

# Use this model to predict on X_test
activations = first_layer_model.predict(X_test)

#%%
import matplotlib.pyplot as plt

# Plot the activations of first digit of X_test for the 15th filter
plt.matshow(activations[0,:,:,14], cmap = 'viridis')

# Do the same but for the 18th filter now
plt.matshow(activations[0,:,:,17], cmap = 'viridis')

plt.show()

"""The 15th filter (a.k.a convolutional mask) learned to detect horizontal traces in your digits. 
On the other hand, filter 18th seems to be checking for vertical traces."""


