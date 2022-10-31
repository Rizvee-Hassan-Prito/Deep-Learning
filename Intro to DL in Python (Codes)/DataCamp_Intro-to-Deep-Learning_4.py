# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 02:13:51 2022

@author: User
"""


# Import necessary modules
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('titanic_all_numeric.csv')

#%%

#Splitting the target and features columns
target = np.asarray(df['survived']).astype('float32')
predictors = df.drop("survived", axis=1).values
predictors = np.asarray(predictors).astype('float32')

#%%

# Convert the target to categorical: target
target = to_categorical(target)
#print(target)

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

input_shape = (n_cols,)

#%%

# Import EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping

# Specify the model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape = input_shape))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
model.fit(predictors, target, validation_split=0.3, epochs=30, callbacks = [early_stopping_monitor])

#%%

import matplotlib.pyplot as plt

model_1=model

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Create the new model: model_2
model_2 = Sequential()

# Add the first and second layers
model_2.add(Dense(100, activation='relu', input_shape=input_shape))

model_2.add(Dense(100, activation='relu'))

# Add the output layer
model_2.add(Dense(2, activation='softmax'))

# Compile model_2
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model_1
model_1_training = model_1.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Fit model_2
model_2_training = model_2.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

