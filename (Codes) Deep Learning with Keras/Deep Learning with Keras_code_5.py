# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:41:57 2022

@author: User
"""

import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

irr_mach = pd.read_csv("irrigation_machine.csv")

target_values = irr_mach[["parcel_0", "parcel_1", "parcel_2"]]

train = irr_mach
train.drop(["Unnamed: 0", "parcel_0", "parcel_1", "parcel_2"], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(train, target_values, test_size = 0.2, random_state = 42)


#%%

def get_model(activation):
    # Instantiate a Sequential model
    model = Sequential()

    # Add a hidden layer of 64 neurons and a 20 neuron's input
    model.add(Dense(64, input_shape=(20,), activation = activation))

    # Add an output layer of 3 neurons with sigmoid activation
    model.add(Dense(3, activation='sigmoid'))
    
    # Compile your model with binary crossentropy loss
    model.compile(optimizer='adam',
               loss = "binary_crossentropy",
               metrics=['accuracy'])
    
    return model

#%%

# Activation functions to try
activations = ['relu', 'leaky_relu', 'sigmoid', 'tanh']

# Loop over the activation functions
activation_results = {}

for act in activations:
  
    # Get a new model with the current activation
  model = get_model(act)
  
  # Fit the model and store the history results
  h_callback = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=0)
  
  activation_results[act] = h_callback
  
  print("\n",activation_results)

#%%

import matplotlib.pyplot as plt

# Extract val_loss history of each activation function 

val_loss_per_function = {k:v.history['val_loss'] for k,v in activation_results.items()}

#print(activation_results.items())

# Extract val_loss history of each activation function 

val_acc_per_function = {k:v.history['val_accuracy'] for k,v in activation_results.items()}

# Create a dataframe from val_loss_per_function
val_loss= pd.DataFrame(val_loss_per_function)

# Call plot on the dataframe
val_loss.plot()
plt.show()

# Create a dataframe from val_acc_per_function
val_acc = pd.DataFrame(val_acc_per_function)

# Call plot on the dataframe
val_acc.plot()
plt.show()

