#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:48:11 2020

@author: christian
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model

import os
#import matplotlib.axis as ax
import matplotlib.pyplot as plt
#from matplotlib.ticker import MaxNLocator

 #%%

PATH = '/media/christian/HardDrive_Christian/Masterthesis/Classifier/Data/Spektogramme/JPG_small'
#PATH = '/home/christian/Documents/Classifier/Data/Spektogramme/JPG' 

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'val')

train_WEA2_dir = os.path.join(train_dir, 'WEA2')  
train_WEA3_dir = os.path.join(train_dir, 'WEA3') 
train_WEA4_dir = os.path.join(train_dir, 'WEA4')
validation_WEA2_dir = os.path.join(validation_dir, 'WEA2') 
validation_WEA3_dir = os.path.join(validation_dir, 'WEA3')  
validation_WEA4_dir = os.path.join(validation_dir, 'WEA4')

#%%

num_WEA2_tr = len(os.listdir(train_WEA2_dir))
num_WEA3_tr = len(os.listdir(train_WEA3_dir))
num_WEA4_tr = len(os.listdir(train_WEA4_dir))

num_WEA2_val = len(os.listdir(validation_WEA2_dir))
num_WEA3_val = len(os.listdir(validation_WEA3_dir))
num_WEA4_val = len(os.listdir(validation_WEA4_dir))

total_train = num_WEA2_tr + num_WEA3_tr +  num_WEA4_tr
total_val = num_WEA2_val + num_WEA3_val + num_WEA4_val

print('total training WEA2 images:', num_WEA2_tr)
print('total training WEA3 images:', num_WEA3_tr)
print('total training WEA4 images:', num_WEA4_tr)

print('total validation WEA2 images:', num_WEA2_val)
print('total validation WEA3 images:', num_WEA3_val)
print('total validation WEA4 images:', num_WEA4_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

#%%

batch_size = 64
epochs = 100
Lr = 0.001
Do=0.2
#20000er
IMG_HEIGHT = 200
IMG_WIDTH = 300

#4000er
#IMG_HEIGHT = 369
#IMG_WIDTH = 495

#%%

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

#%%

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical',
                                                           color_mode='grayscale'
                                                           )

#%%

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical',
                                                              color_mode='grayscale'
                                                              )

#%%

sample_training_images, _ = next(train_data_gen)

#%%

#print(train_data_gen.class_indices)

#%%
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
#def plotImages(images_arr):
#    fig, axes = plt.subplots(1, 5, figsize=(20,20))
#    axes = axes.flatten()
#    for img, ax in zip( images_arr, axes):
#        ax.imshow(img)
#        ax.axis('off')
#    plt.tight_layout()
#    plt.show()
#
#%%
#    
#plotImages(sample_training_images[:2])

#%%  

#model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

model = Sequential([
    Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH ,1)),
#    Dense(4096, activation='relu'),
#    Dropout(Do),
    Dense(1024, activation='relu'),
    Dropout(Do),
    Dense(512, activation='relu'),
    Dropout(Do),
    Dense(128, activation='relu'),
    Dense(3)
])
    
#%%

#model.compile(optimizers.SGD(lr=0.0005, decay=1e-6),
#              loss='mean_squared_error',
#              metrics=['accuracy'])

model.compile(optimizers.Adam(lr=Lr),
              loss='mean_squared_error',
              metrics=['accuracy'])

#%%

print('Batch Size: ', batch_size)
print('Epochs: ', epochs)
print('Dropout: ', Do)
print('Lernrate: ', Lr)
print('Optimizer: Adam')
print('Loss: mean_squared_error')


model.summary()

#plot_model(model, to_file='FCN_Graph.png', show_shapes=True, expand_nested=True)



#%%

history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    verbose=2,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

#%%

#confusion = tf.confusion_matrix(labels=y_, predictions=y, num_classes=num_classes)
#print(confusion)

#%%

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']


loss=history.history['loss']
val_loss=history.history['val_loss']

print('train acc: ', acc,'vall acc: ', val_acc,'train loss: ', loss,'val loss: ', val_loss)

epochs_range = range(epochs)

plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, color='greenyellow', marker='o', markerfacecolor='green', linewidth=2, label='Training')
plt.plot(epochs_range, val_acc, color='skyblue', marker='D', markerfacecolor='blue', linewidth=2, label='Validiation')
plt.legend()
#ax.XAxis.set_major_locator(MaxNLocator(integer=True))
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch no.')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, color='greenyellow', marker='o', markerfacecolor='green', linewidth=2, label='Training')
plt.plot(epochs_range, val_loss, color='skyblue', marker='D', markerfacecolor='blue', linewidth=2, label='Validiation')
plt.legend()
#ax.XAxis.set_major_locator(MaxNLocator(integer=True))
plt.title('Training and Validation Loss')
plt.xlabel('Epoch no.')
plt.ylabel('Loss')
plt.savefig('Hist.png')