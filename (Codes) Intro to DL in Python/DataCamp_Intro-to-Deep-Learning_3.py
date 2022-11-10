# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 21:24:52 2022

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

#%%

from tensorflow.keras.optimizers import SGD

def get_new_model(input_shape):
    # Set up the model
    model = Sequential()

    # Add the first layer
    model.add(Dense(100,activation='relu', input_shape = (input_shape,)))
    
    model.add(Dense(100,activation='relu'))
    
    # Add the output layer
    model.add(Dense(2,activation='softmax'))
    
    return (model)


# Create list of learning rates: lr_to_test
lr_to_test = [0.000001, 0.01, 1]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    # Build new model to test, unaffected by previous models
    model = get_new_model(n_cols)
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr=lr)
    
    # Compile the model
    model.compile(optimizer=my_optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
    
    # Fit the model
    model.fit(predictors,target)

