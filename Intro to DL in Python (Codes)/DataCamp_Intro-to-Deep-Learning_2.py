# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 02:59:54 2022

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

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(32,activation='relu',input_shape=(n_cols,)))

# Add the output layer
model.add(Dense(2,activation='softmax'))

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

# Fit the model
model.fit(predictors,target)


#%%

from tensorflow.keras.models import load_model

# Save model
model.save('model_Titanic.h5')

# Load model
my_model = load_model('model_Titanic.h5')

# Predicting training data with the loaded model
predictions = my_model.predict(predictors)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:,1]

# Print predicted_prob_true
print(predicted_prob_true)
