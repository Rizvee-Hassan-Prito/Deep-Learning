# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 15:48:12 2022

@author: User
"""

import tensorflow as tf
import matplotlib.pyplot as plt 

mnist= tf.keras.datasets.mnist

(x_train,y_train), (x_test,y_test)= mnist.load_data()
#%%

plt.imshow(x_train[0])
#plt.imshow(x_train[2])
#plt.imshow(x_train[5])

#plt.imshow(x_train[0],cmap=plt.cm.binary)

#%%
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

plt.imshow(x_train[0])

#%%

model= tf.keras.models.Sequential() #equential()-->Feed-forward the img
model.add(tf.keras.layers.Flatten()) #Input layer
model.add(tf.keras.layers.Dense(128, activation= tf.nn.relu)) # Hidden layer, 128-->neurons, relu-->activation functions
model.add(tf.keras.layers.Dense(128, activation= tf.nn.relu)) # 2nd Hidden Layer
model.add(tf.keras.layers.Dense(10, activation= tf.nn.softmax)) # Output Layer, 10--> number of classes
                                                                # softmax-->for probability distribution
model.compile(optimizer='adam' ,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                                                    #loss--> for classifying cats or dogs, it would be "binary" 
model.fit(x_train,y_train, epochs=3)

#%%

val_loss, val_acc= model.evaluate(x_test,y_test)
print(val_loss, val_acc)


#%%

model.save("Number_Reader.model") #saving the model
new_model= tf.keras.models.load_model("Number_Reader.model") # Loading the model

#%%

prediction=new_model.predict([x_test])
print(prediction) #Printing one-hot encoded arrays of probability distributions

#%%

import numpy as np

print(np.argmax(prediction[0]))

plt.imshow(x_test[0])
#plt.show()

