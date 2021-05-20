#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:15:53 2020

@author: christian
"""
import random
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D, Reshape
from tensorflow.keras import optimizers
#from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt


#%%

batch_size = 64
epochs = 10
Lr = 0.001
Do=0.2

#%%

def load_samples(csv_file):
    data = pd.read_csv(os.path.join('/media/christian/HardDrive_Christian/Masterthesis/Classifier/Data/zeitreihen/time-series-cutted/data_files',csv_file))
    data = data[['FileName', 'Label', 'ClassName']]
    file_names = list(data.iloc[:,0])
    # Get the labels present in the second column
    labels = list(data.iloc[:,1])
    samples=[]
    for samp,lab in zip(file_names,labels):
        samples.append([samp,lab])
    return samples
samples = load_samples('ts_recognition_train.csv')

#%%

def generator(samples, batch_size=batch_size,shuffle_data=True):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        random.shuffle(samples)

        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = samples[offset:offset+batch_size]

            # Initialise X_train and y_train arrays for this batch
            X_train = []
            y_train = []

            # For each example
            for batch_sample in batch_samples:
                # Load image (X) and label (y)
                ts_name = batch_sample[0]
                label = batch_sample[1]
                
                with open(ts_name, 'r') as f:
                    data = f.read().split()
                    ts = []
                    for elem in data:
                        try:
                            ts.append(float(elem))
                        except ValueError:
                            pass
                    #print(ts[:3])
                                                
                # Add example to arrays
                X_train.append(ts)
                y_train.append(label)

            # Make sure they're numpy arrays (as opposed to lists)
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # The generator-y part: yield the next training batch            
            yield X_train, y_train
            
#%%
            
# this will create a generator object
train_datagen = generator(samples,batch_size=batch_size,shuffle_data=True)
x,y = next(train_datagen)
print (x.shape)
#output: (8, 224, 224, 3)
print (y)
#output: [0 1 1 4 3 1 4 2]

#%%

# Import list of train and validation data (image filenames and image labels)
train_samples = load_samples('ts_recognition_train.csv')
validation_samples = load_samples('ts_recognition_val.csv')

# Create generator
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

#%%  

#model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

model = Sequential([
    Reshape((240000,1), input_shape=(240000,)),
    Conv1D(16, 5, padding='same', activation='relu', input_shape=(240000,1)),
    MaxPooling1D(),
    Dropout(Do),
    Conv1D(32, 5, padding='same', activation='relu'),
    MaxPooling1D(),
    Conv1D(64, 5, padding='same', activation='relu'),
    MaxPooling1D(),
    Flatten(),
    Dense(512, activation='relu'),
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

#plot_model(model, to_file='CNN_Graph_3Conv2Dense.png', show_shapes=True, expand_nested=True)


#%%

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_samples) // batch_size,
    epochs=epochs,
    verbose=2,
    validation_data=validation_generator,
    validation_steps=len(validation_samples) // batch_size
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