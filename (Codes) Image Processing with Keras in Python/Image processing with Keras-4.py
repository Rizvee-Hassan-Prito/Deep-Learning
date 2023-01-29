# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 18:35:33 2023

@author: User
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# load dataset
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

#print(train_data.shape)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()

# Add a convolutional layer (15 units)
model.add(Conv2D(20, kernel_size=2, activation='relu', input_shape=(28, 28, 1)))

# Add a pooling operation
model.add(MaxPool2D(2))

# Add another convolutional layer (5 units)
model.add(Conv2D(20, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
#%%
# Train the model and store the training object
training = model.fit(train_data, train_labels, validation_split=0.2, epochs=3, 
                     batch_size=10)

# Extract the history from the training object
history = training.history

# Plot the training loss 
plt.plot(history['loss'])
# Plot the validation loss
plt.plot(history['val_loss'])

plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(["Training", "Validation"])

# Show the figure
plt.show()
#%%

from keras.callbacks import ModelCheckpoint

# This checkpoint object will store the model parameters in the file "weights.hdf5"

checkpoint = ModelCheckpoint('weights.hdf5', 
                             monitor='val_loss',save_best_only=True)

# Store in a list to be used during training
callbacks_list = [checkpoint]

print(callbacks_list[0])

# Fit the model on a training set, using the checkpoint as a
#callback
training = model.fit(train_data, train_labels, validation_split=0.2, 
          epochs=3, callbacks=callbacks_list)

# Extract the history from the training object
history = training.history

# Plot the training loss 
plt.plot(history['loss'])
# Plot the validation loss
plt.plot(history['val_loss'])

plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(["Training", "Validation"])

# Show the figure
plt.show()

#%%
# Load the weights from file
model.load_weights('weights.hdf5')

# Predict from the first three images in the test data
prediction = model.predict(test_data[:3])

print(np.argmax(prediction, axis=1))

print(callbacks_list[0].best)

#%%

from keras.layers import Dropout

model = Sequential()

# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu', 
                 input_shape=(28, 28, 1)))

# Add a dropout layer applying on the first layer
model.add(Dropout(0.2))

# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

#%%

from keras.layers import BatchNormalization

model = Sequential()

# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu', 
                 input_shape=(28, 28, 1)))


# Add batch normalization layer
model.add(BatchNormalization())

# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

#%%

model = Sequential()

# Add a convolutional layer (15 units)
model.add(Conv2D(20, kernel_size=2, activation='relu', input_shape=(28, 28, 1)))

# Add a pooling operation
model.add(MaxPool2D(2))

# Add another convolutional layer (5 units)
model.add(Conv2D(20, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Get the first convolutional layer from the model
c1 = model.layers[0]

# Get the weights of the first convolutional layer
weights1 = c1.get_weights()

kernels = weights1[0]

# Pull out the first channel of the first kernel in the first layer
kernel_1_1=kernels[:,:,0,1]

print(kernel_1_1)

plt.imshow(kernel_1_1)
#%%

def convolutional (im,  kernel):
    
    result = np.zeros(im.shape)

    # Output array
    for ii in range(im.shape[0]-2):
        for jj in range(im.shape[1]-2):
            result[ii, jj] = (im[ii:ii+2, jj:jj+2] * kernel).sum()

    return result

test_image = test_data[4]
plt.imshow(test_image)
plt.show()

filtered_image = convolutional(test_image, kernel_1_1)
plt.imshow(filtered_image)


