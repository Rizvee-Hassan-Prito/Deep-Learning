# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 02:55:07 2022

@author: User
"""

#%%
"""
ResNet50 is a model trained on the Imagenet dataset that is able to distinguish between 1000 different labeled objects.
ResNet50 is a deep model with 50 layers, you can check it in 3D here(https://tensorspace.org/html/playground/resnet50.html).
"""

"""
The original ResNet50 model was trained with images of size 224 x 224 pixels and a number of preprocessing operations; like the subtraction of the mean pixel value in the training set for all training images. You need to pre-process the images you want to predict on in the same way.
When predicting on a single image you need it to fit the model's input shape, which in this case looks like this: (batch-size, width, height, channels),np.expand_dims with parameter axis = 0 adds the batch-size dimension, representing that a single image will be passed to predict. This batch-size dimension value is 1, since we are only predicting on one image.
You will go over these preprocessing steps as you prepare this dog's (named Ivy) image into one that can be classified by ResNet50.
"""

#%%
import numpy as np
import pandas as pd

# Import image and preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

img_path='D:\Courses\Deep Learning (Self-learning)\DataCamp\(Codes) Deep Learning with Keras\dog.png'

# Load the image with the right target size for your model
img = image.load_img(img_path, target_size=(224, 224))

# Turn it into an array
img_array = image.img_to_array(img)

# Expand the dimensions of the image, this is so that it fits the expected model input format
img_expanded = np.expand_dims(img_array, axis = 0)

# Pre-process the img in the same way original images were
img_ready = preprocess_input(img_expanded)

#%%
import matplotlib.pyplot as plt

print(img_ready.shape) 
#plt.matshow(img_ready[0,:,:,0], cmap = 'viridis')
plt.imshow(img_ready[0,:,:,0])

#%%

from tensorflow.keras.applications.resnet50 import ResNet50 , decode_predictions

# Instantiate a ResNet50 model with 'imagenet' weights
model = ResNet50(weights='imagenet')

# Predict with ResNet50 on your already processed img
preds = model.predict(img_ready)

# Decode the first 3 predictions
print('Predicted:', decode_predictions(preds, top=3)[0])
