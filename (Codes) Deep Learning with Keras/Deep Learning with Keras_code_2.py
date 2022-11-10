# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 22:36:00 2022

@author: User
"""
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Instantiate a Sequential model
model = Sequential()

# Add a hidden layer of 64 neurons and a 20 neuron's input
model.add(Dense(64, input_shape=(20,), activation='relu'))

# Add an output layer of 3 neurons with sigmoid activation
model.add(Dense(3, activation='sigmoid'))

# Compile your model with binary crossentropy loss
model.compile(optimizer='adam',
           loss = "binary_crossentropy",
           metrics=['accuracy'])

model.summary()

#%%

from sklearn.model_selection import train_test_split

irr_mach = pd.read_csv("irrigation_machine.csv")

target_values = irr_mach[["parcel_0", "parcel_1", "parcel_2"]]

train = irr_mach
train.drop(["Unnamed: 0", "parcel_0", "parcel_1", "parcel_2"], axis=1, inplace=True)

sensors_train, sensors_test, parcels_train, parcels_test = train_test_split(train, target_values, test_size = 0.2, random_state = 42)

#%%

# Train for 100 epochs using a validation split of 0.2
model.fit(sensors_train, parcels_train, epochs = 100, validation_split = 0.2)

# Predict on sensors_test and round up the predictions
preds = model.predict(sensors_test)
preds_rounded = np.round(preds)

# Print rounded preds
print('Rounded Predictions: \n', preds_rounded)

# Evaluate your model's accuracy on the test data
accuracy = model.evaluate(sensors_test,parcels_test)[1]

# Print accuracy
print('Accuracy:', accuracy)
