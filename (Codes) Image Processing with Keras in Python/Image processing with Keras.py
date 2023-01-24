# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 21:14:13 2023

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np

data = plt.imread('sign.jpg')
print(data.shape)

# Display the image
plt.imshow(data)
plt.show()

#%%
data=np.array(data)

# Set the red channel in this part of the image to 1
data[:300,:300,0] = 400

# Set the green channel in this part of the image to 0
data[:300,:300,1] = 0

# Set the blue channel in this part of the image to 0
data[:300,:300,2] = 0

# Visualize the result
plt.imshow(data)
plt.show()

#%%

labels=np.array(['shoe', 'shirt', 'shoe', 'shirt', 'dress', 'dress', 'dress', 'shirt'])

# The number of image categories
n_categories = 3

# The unique values of categories in the data
categories = np.array(["shirt", "dress", "shoe"])

# Initialize ohe_labels as all zeros
one_labels = np.zeros((len(labels), n_categories))

# Loop over the labels
for ii in range(len(labels)):
    # Find the location of this label in the categories variable
    jj = np.where(categories==labels[ii])
    # Set the corresponding zero to one
    one_labels[ii,jj] = 1

print(one_labels)

#%%

"""
test_labels = np.array([[0., 0., 1.],
                       [0., 1., 0.],
                       [0., 0., 1.],
                       [0., 1., 0.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 1., 0.]])
"""

predictions = np.array([[0., 0., 1.],
                       [0., 1., 0.],
                       [0., 0., 1.],
                       [1., 0., 0.],
                       [0., 0., 1.],
                       [1., 0., 0.],
                       [0., 0., 1.],
                       [0., 1., 0.]])

# Calculate the number of correct predictions
number_correct = (one_labels*predictions).sum()
print(number_correct)

# Calculate the proportion of correct predictions
proportion_correct = number_correct/len(one_labels)
print(proportion_correct)

#%%

# Imports components from Keras
from keras.models import Sequential
from keras.layers import Dense

# Initializes a sequential model
model = Sequential()

# First layer
model.add(Dense(10, activation='relu', input_shape=(784,)))

# Second layer
model.add(Dense(10, activation='relu'))

# Output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', 
           loss='categorical_crossentropy', 
           metrics=['accuracy'])

#%%

from keras.datasets import fashion_mnist
from keras.utils import to_categorical

# load dataset
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

# plot the sample
fig = plt.figure
plt.imshow(train_data[0],)
plt.show()

# specify the number of rows and columns
num_row = 3
num_col = 4

# get a segment of the dataset
num = num_row*num_col
images = train_data[:num]
labels = train_labels[:num]

# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num_row*num_col):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i],)
    ax.set_title('Label: {}'.format(labels[i]))
plt.tight_layout()
plt.show()

#%%

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Reshape the data to two-dimensional array
train_data = train_data.reshape((60000, 784))

# Fit the model
model.fit(train_data, train_labels, validation_split=0.2, epochs=70, verbose=1)

#%%

# Reshape test data
test_data = test_data.reshape((10000, 784))

# Evaluate the model
model.evaluate(test_data, test_labels)

prediction=model.predict(test_data)

print(np.argmax(test_labels, axis=1))
print(np.argmax(prediction, axis=1))
