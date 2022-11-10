# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 23:20:01 2022

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
import matplotlib.pyplot as plt


irr_mach = pd.read_csv("irrigation_machine.csv")

target_values = irr_mach[["parcel_0", "parcel_1", "parcel_2"]]

train = irr_mach
train.drop(["Unnamed: 0", "parcel_0", "parcel_1", "parcel_2"], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(train, target_values, test_size = 0.2, random_state = 42)

# Train your model and save its history
h_callback = model.fit(X_train, y_train, epochs = 25,
               validation_data=(X_test, y_test))

def plot_loss(loss,val_loss):
  # Plot training & validation loss values
  plt.figure()
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()

def plot_accuracy(acc,val_acc):
  # Plot training & validation accuracy values
  plt.figure()
  plt.plot(acc)
  plt.plot(val_acc)
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()

# Plot train vs test loss during training
plot_loss(h_callback.history["loss"], h_callback.history["val_loss"])

# Plot train vs test accuracy during training
plot_accuracy(h_callback.history["accuracy"], h_callback.history["val_accuracy"])

#%%

# Import the early stopping callback
from tensorflow.keras.callbacks import EarlyStopping

# Define a callback to monitor val_accuracy
monitor_val_acc = EarlyStopping(monitor="val_accuracy", 
                       patience=5)

# Train your model using the early stopping callback
model.fit(X_train, y_train, 
           epochs=1000, validation_data=(X_test, y_test),
           callbacks = [monitor_val_acc])

#%%

""""Deep learning models can take a long time to train, especially when you move to deeper architectures and bigger datasets. 
Saving your model every time it improves as well as stopping it when it no longer does allows you to worry less about choosing the number of epochs to train for. 
You can also restore a saved model anytime and resume training where you left it."""

# Import the EarlyStopping and ModelCheckpoint callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Early stop on validation accuracy
monitor_val_acc = EarlyStopping(monitor = "val_accuracy", patience = 3)

# Save the best model as best_banknote_model.hdf5
model_checkpoint = ModelCheckpoint("irrigation_machine_model.hdf5", save_best_only = True)

# Fit your model for a stupid amount of epochs
h_callback = model.fit(X_train, y_train,
                    epochs = 1000000000000,
                    callbacks = [monitor_val_acc,model_checkpoint],
                    validation_data = (X_test, y_test))
