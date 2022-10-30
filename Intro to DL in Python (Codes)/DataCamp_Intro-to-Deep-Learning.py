# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 01:53:55 2022

@author: User
"""

# Import necessary modules
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

predictors = pd.read_csv('hourly_wages.csv')


#%%

#Splitting the target and features columns
target = np.array(predictors['wage_per_hour'])
predictors = predictors.drop("wage_per_hour", axis=1)
predictors = np.array(predictors)

#%%
# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32, activation='relu', input_shape=((n_cols,))))

# Add the output layer
model.add(Dense(1))


#%%
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss)

#Fit the model
model.fit(predictors,target)

#%%


