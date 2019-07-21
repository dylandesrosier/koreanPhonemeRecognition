from __future__ import print_function

import keras
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.metrics import accuracy_score

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
import os

import parseData

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'phoneme_recognition_model.h5'

data, labels = parseData.iterateOverData()
train_data = np.expand_dims(data, -1)
train_labels = to_categorical(labels, num_classes=25)
train_data = np.array(train_data)
train_labels = np.array(train_labels)


# Tweak this depending on what your data looks like
# def prepare_data():
#     train_data, train_labels, test_data, test_labels = [], [], [], []
#     is_first = True
#     with open('fer2013.csv', 'r') as f:
#         for line in f.readlines():
#             if is_first:
#                 is_first = False
#                 continue
#             label, data, example_type = line.split(',')
#             formatted_data = np.array([int(x) for x in data.split()]).reshape(48, 48)
#             normalized_data = formatted_data.astype('float32')/255.0
#             train_data.append(normalized_data)
#             train_labels.append(label)
#     f.close()
#     test_data = np.expand_dims(test_data, -1)
#     test_labels = to_categorical(test_labels, num_classes=7)
#     train_data = np.expand_dims(train_data, -1)
#     train_labels = to_categorical(train_labels, num_classes=7)
#     return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)

# train_data, train_labels, test_data, test_labels = prepare_data()

epochs = 10
num_classes = 25 # How many phonemes are there?
batch_size = 1

train_shape = train_data[0].shape

model = Sequential()
# Change first 2 numbers depending on train shape
# Num 1 is the number of neurons in this layer, ie the number of filters you're going to learn
# Num 2 is the size of the convolution
model.add(Conv1D(32, 1, input_shape=train_shape, kernel_regularizer=l2(0.01), activation='relu'))
model.add(Conv1D(32, 1, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Normally, you'd pool here with model.add(MaxPooling2D(pool_size=(2, 2))) but I don't think we'll need any of those
model.add(MaxPooling1D(pool_size=1))
model.add(Dropout(0.5)) # Just for fun, let's use some dropout

model.add(Flatten())
model.add(Dense(32, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Softmax if you want all the probabilities to add up to 1
# Sigmoid if you want independent probabilities that the label of that example is each class
model.add(Dense(num_classes, activation='sigmoid'))

# Adam optimizer is standard (Controls LR as epochs progress)
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())

learning_rate_adapter = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', min_lr=0.0001)


model.fit(train_data, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1,
          shuffle=True)

# Data augmentation?

# Save the model!
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
# scores = model.evaluate(test_data, test_labels, verbose=1)

# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
