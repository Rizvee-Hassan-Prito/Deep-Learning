# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 02:22:34 2022

@author: User
"""

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_csv("breast-cancer.csv")

df.rename(columns = {'diagnosis':'target'}, inplace = True)

# Transform into a categorical variable
df.target = pd.Categorical(df.target)

# Assign a number to each category (label encoding)
df.target = df.target.cat.codes

# Print the label encoded competitors
print('Label encoded competitors: \n',df.target.head())

# Import to_categorical from keras utils module
from tensorflow.keras.utils import to_categorical

# Use to_categorical on your labels
target = to_categorical(df.target)

# Now print the one-hot encoded labels
print('One-hot encoded competitors: \n',target)

#%%
from sklearn.model_selection import train_test_split

target_values = target
train = df
train.drop("target", axis=1, inplace=True)
train.drop("id", axis=1, inplace=True)
train.drop("Unnamed: 32", axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(train, target_values, test_size = 0.3, random_state = 42)

#%%

from tensorflow.keras.optimizers import Adam

# Creates a model given an activation and learning rate
def create_model(learning_rate, activation):
  
  	# Create an Adam optimizer with the given learning rate
  	opt = Adam(learning_rate = learning_rate)
  	
  	# Create your binary classification model  
  	model = Sequential()
  	model.add(Dense(128, input_shape = (30,), activation = activation))
  	model.add(Dense(256, activation = activation))
  	model.add(Dense(2, activation = 'sigmoid'))
  	
  	# Compile your model with your optimizer, loss, and metrics
  	model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
  	return model

#%%

from sklearn.model_selection import RandomizedSearchCV

# Import KerasClassifier from tensorflow.keras scikit learn wrappers
from tensorflow.keras.wrappers.scikit_learn  import KerasClassifier 

# Create a KerasClassifier
model = KerasClassifier(build_fn = create_model)

# Define the parameters to try out
params = {'activation': ['relu', 'tanh'], 'batch_size': [32, 128, 256], 
          'epochs': [50, 100, 200], 'learning_rate': [0.1, 0.01, 0.001]}

# Create a randomize search cv object passing in the parameters to try
random_search = RandomizedSearchCV(model, param_distributions = params, cv = 3)

random_search.fit(X_train,y_train)

# Print results
print("Best Score: ",random_search.best_score_, "\nBest Parameters: ", random_search.best_params_)