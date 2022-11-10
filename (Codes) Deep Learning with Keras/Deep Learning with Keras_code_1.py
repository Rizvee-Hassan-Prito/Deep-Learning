# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 23:52:57 2022

@author: User
"""

# Import necessary modules
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Instantiate a sequential model
model = Sequential()
  
# Add 3 dense layers of 128, 64 and 32 neurons each
model.add(Dense(128, input_shape=(2,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
  
# Add a dense layer with as many neurons as competitors
model.add(Dense(4, activation='softmax'))
  
# Compile your model using categorical_crossentropy loss
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#%%

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

coord_train, coord_test, competitors_train, competitors_test = train_test_split(train, target_values, test_size = 0.2, random_state = 42)


# Fit your model to the training data for 200 epochs
model.fit(coord_train,competitors_train,epochs=200)

# Evaluate your model accuracy on the test data
accuracy = model.evaluate(coord_test, competitors_test)

# Print accuracy
print('Accuracy:', accuracy[1])

#%%

#Making small testing dataset by randomly choosing rows from testing part of our dataset

coords_small_test=result = coord_test.iloc[[23,53,34,61,75]]

competitors_small_test=[]
competitors_small_test.append(competitors_test[23])
competitors_small_test.append(competitors_test[53])
competitors_small_test.append(competitors_test[34])
competitors_small_test.append(competitors_test[61])
competitors_small_test.append(competitors_test[75])

competitors_small_test=np.array(competitors_small_test)


# Predict on coords_small_test
preds = model.predict(coords_small_test)

# Print preds vs true values
print("{:45} | {}".format('Raw Model Predictions','True labels'))
for i,pred in enumerate(preds):
  print("{} | {}".format(pred,competitors_small_test[i]))
  
#%%

# Predict on coords_small_test
preds = model.predict(coords_small_test)

# Print preds vs true values
print("{:45} | {}".format('Raw Model Predictions','True labels'))
for i,pred in enumerate(preds):
  print("{} | {}".format(pred,competitors_small_test[i]))

# Extract the position of highest probability from each pred vector
preds_chosen = [np.argmax(pred) for pred in preds]

# Print preds vs true values
print("{:10} | {}".format('Rounded Model Predictions','True labels'))
for i,pred in enumerate(preds_chosen):
  print("{:25} | {}".format(pred,competitors_small_test[i]))