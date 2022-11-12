# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 19:39:56 2022

@author: User
"""

import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

"""Autoencoders have several interesting applications like anomaly detection or image denoising. They aim at producing an output identical to its inputs.
 The input will be compressed into a lower dimensional space, encoded. The model then learns to decode it back to its original form."""


"""You will encode and decode the MNIST dataset of handwritten digits, the hidden layer will encode a 32-dimensional representation of the image, which originally consists of 784 pixels (28 x 28). 
The autoencoder will essentially learn to turn the 784 pixels original image into a compressed 32 pixels image and
 learn how to use that encoded representation to bring back the original 784 pixels image."""


import tensorflow as tf
import matplotlib.pyplot as plt 

mnist= tf.keras.datasets.mnist

(x_train,y_train), (x_test,y_test)= mnist.load_data()

#%%

# Start with a sequential model
autoencoder = Sequential()

# Add a dense layer with input the original image pixels and neurons the encoded representation
autoencoder.add(Dense(32, input_shape=(784, ), activation="relu"))

# Add an output layer with as many neurons as the orginal image pixels
autoencoder.add(Dense(784, activation = "sigmoid"))

# Compile your model with adadelta
autoencoder.compile(optimizer = "adadelta", loss = "binary_crossentropy")

# Summarize your model structure
autoencoder.summary()

#%%
"""
# Build your encoder by using the first layer of your autoencoder
encoder = Sequential()
encoder.add(autoencoder.layers[0])

# Encode the noisy images and show the encodings for your favorite number [0-9]
encodings = encoder.predict(X_test_noise)
show_encodings(encodings, number = 1)

# Predict on the noisy images with your autoencoder
decoded_imgs = autoencoder.predict(X_test_noise)

# Plot noisy vs decoded images
compare_plot(X_test_noise, decoded_imgs)
"""