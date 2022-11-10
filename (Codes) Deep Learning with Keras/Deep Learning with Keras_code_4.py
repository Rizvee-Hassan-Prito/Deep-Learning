# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 01:03:15 2022

@author: User
"""

import numpy as np
import pandas as pd

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

darts = pd.read_csv("darts.csv")

# Transform into a categorical variable
darts.competitor = pd.Categorical(darts.competitor)

# Assign a number to each category (label encoding)
darts.competitor = darts.competitor.cat.codes

# Print the label encoded competitors
print('Label encoded competitors: \n',darts.competitor.head())

# Import to_categorical from keras utils module
from tensorflow.keras.utils import to_categorical

coordinates = darts.drop(['competitor'], axis=1)

# Use to_categorical on your labels
competitors = to_categorical(darts.competitor)

# Now print the one-hot encoded labels
print('One-hot encoded competitors: \n',competitors)

#%%
from sklearn.model_selection import train_test_split

target_values = competitors
train = darts
train.drop("competitor", axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(train, target_values, test_size = 0.3, random_state = 42)

#%%
# Instantiate a Sequential model
model = Sequential()

# Input and hidden layer with input_shape, 16 neurons, and relu 
model.add(Dense(16, input_shape = (2,), activation = "relu"))

# Output layer with 10 neurons (one per digit) and softmax
model.add(Dense(4, activation = 'softmax'))

# Compile your model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Test if your model is well assembled by predicting before training
print(model.predict(X_train))

#%%
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Store initial model weights
init_weights = model.get_weights()

# Lists for storing accuracies
train_accs = []
test_accs = []

training_sizes = [ 125, 250,  390, 485]

for size in training_sizes:
  	# Get a fraction of training data (we only care about the training data)
    X_train_frac, y_train_frac = X_train[:size], y_train[:size]

    # Reset the model to the initial weights and train it on the new training data fraction
    
    model.set_weights(init_weights)
    
    model.fit(X_train_frac, y_train_frac, epochs = 50, callbacks = [EarlyStopping(monitor='loss'
                                                                                  , patience=2)])

    # Evaluate and store both: the training data fraction and the complete test set results
    train_accs.append(model.evaluate(X_train, y_train)[1])
    test_accs.append(model.evaluate(X_test, y_test)[1])
    
print(train_accs)
print(test_accs)
