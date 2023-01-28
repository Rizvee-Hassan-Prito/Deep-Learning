# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 14:36:37 2023

@author: User
"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

# load dataset
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

#print(train_data.shape)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()

# Add a convolutional layer (15 units)
model.add(Conv2D(15, kernel_size=2, activation='relu', input_shape=(28, 28, 1)))

# Add another convolutional layer (5 units)
model.add(Conv2D(5, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Fit the model to training data 
model.fit(train_data, train_labels, 
          validation_split=0.2, 
          epochs=3, batch_size=10)

# Evaluate the model on test data
model.evaluate(test_data, test_labels, batch_size=10)

#%%
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as image

data= image.open('sign.jpg')
data= data.convert('1', dither=image.NONE)

plt.imshow(data)
plt.show()
"""
#%%

# CNN model
model = Sequential()
model.add(Conv2D(10, kernel_size=2, activation='relu', 
                 input_shape=(28, 28, 1)))
model.add(Conv2D(10, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

# Summarize the model 
model.summary()

#%%

import matplotlib.pyplot as plt
import numpy as np

im=plt.imread('bricks.jpg')

# Result placeholder
result = np.zeros((im.shape[0]//2, im.shape[1]//2))

# Pooling operation
for ii in range(result.shape[0]):
    for jj in range(result.shape[1]):
        result[ii, jj] = np.max(im[ii*2:ii*2+2, jj*2:jj*2+2])
        
print(result)

#%%

from keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model=Sequential()

# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu', 
                 input_shape=(28, 28, 1)))

# Add a pooling operation
model.add(MaxPool2D(2))

# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

#%%

from keras.datasets import fashion_mnist
from keras.utils import to_categorical

# load dataset
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

#print(train_data.shape)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit to training data
model.fit(train_data, train_labels, 
          validation_split=0.2, 
          epochs=3, batch_size=10)

# Evaluate on test data 
model.evaluate(test_data, test_labels, batch_size=10)