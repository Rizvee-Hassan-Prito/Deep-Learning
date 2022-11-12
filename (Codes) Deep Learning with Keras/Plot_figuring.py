# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 01:44:29 2022

@author: User
"""

import numpy as np

list1 =[ [ [ [1,2,3],[4,5,6]] ,[ [7,8,9],[10,12,13] ], [ [14,15,16],[21,22,23]] ,[ [34,35,36],[41,42,43] ] ], [ [ [54,55,56],[61,62,63]] ,[ [74,75,76],[81,82,83] ], [ [94,95,96],[91,92,93]] ,[ [104,105,106],[111,112,113] ] ] ]

list1= np.array(list1)

print(list1)

print(list1[0,1,:,2])

print(list1[0,:,:,2])

print(list1[0,:,1,:])

#%%
import matplotlib.pyplot as plt

# Plot the activations of first digit of X_test for the 15th filter
plt.matshow(list1[0,:,:,2], cmap = 'viridis')

# Do the same but for the 18th filter now
plt.matshow(list1[0,:,1,:], cmap = 'viridis')

plt.show()