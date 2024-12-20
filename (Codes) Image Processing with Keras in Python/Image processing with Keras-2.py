# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 19:06:49 2023

@author: User
"""

import numpy as np

array = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
kernel = np.array([1, -1, 0])
conv = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Output array
for ii in range(8):
    conv[ii] = (kernel * array[ii:ii+3]).sum()

# Print conv
print(conv)

#%%
import matplotlib.pyplot as plt
import numpy as np

im = plt.imread('bricks.jpg')

kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
result = np.zeros(im.shape)

# Output array
for ii in range(im.shape[0] - 3):
    for jj in range(im.shape[1] - 3):
        result[ii, jj] = (im[ii:ii+3, jj:jj+3] * kernel).sum()

# Print result
print(result)

#%%

# Define a kernel that finds horizontal lines in images.
kernel = np.array([[-1, -1, -1], 
                   [1, 1, 1],
                   [-1, -1, -1]])

# Define a kernel that finds vertical lines in images.
kernel = np.array([[-1, 1, -1], 
                   [-1, 1, -1],
                   [-1, 1, -1]])

# Define a kernel that finds a light spot surrounded by dark pixels.
kernel = np.array([[-1, -1, -1], 
                   [-1, 1, -1],
                   [-1, -1, -1]])

# Define a kernel that finds a dark spot surrounded by bright pixels.
kernel = np.array([[1, 1, 1], 
                   [1, -1, 1],
                   [1, 1, 1]])

#%%

# Import the necessary components from Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

# load dataset
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

#print(train_data.shape)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Initialize the model object
model = Sequential()

# Add a convolutional layer
model.add(Conv2D(10, kernel_size=3, activation='relu', 
               input_shape=(28,28,1)))

# Flatten the output of the convolutional layer
model.add(Flatten())
# Add an output layer for the 3 categories
model.add(Dense(10, activation='softmax'))

# Compile the model 
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

"""Fit the network on train_data and train_labels. Train for 3 epochs with a batch size of 10 images. 
In training, set aside 20% of the data as a validation set, using the validation_split keyword argument."""

# Fit the model on a training set
model.fit(train_data, train_labels, 
          validation_split=0.2, 
          epochs=5, batch_size=10)

# Evaluate the model on separate test data
model.evaluate(test_data, test_labels, batch_size=10)

#%%

# Initialize the model
model = Sequential()

# Add the convolutional layer
model.add(Conv2D(10, kernel_size=3, activation='relu', 
                 input_shape=(28, 28, 1), 
                 padding='same', strides=2))

# Feed into output layer
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Compile the model 
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

"""Fit the network on train_data and train_labels. Train for 3 epochs with a batch size of 10 images. 
In training, set aside 20% of the data as a validation set, using the validation_split keyword argument."""

# Fit the model on a training set
model.fit(train_data, train_labels, 
          validation_split=0.2, 
          epochs=5, batch_size=10)

# Evaluate the model on separate test data
model.evaluate(test_data, test_labels, batch_size=10)