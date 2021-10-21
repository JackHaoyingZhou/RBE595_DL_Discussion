# Carlos Morato, PhD.
# cwmorato@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# Large CNN model for the CIFAR-10 Dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import backend as K
K.set_image_data_format('channels_first')

import numpy
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
import matplotlib.pyplot as plt
from keras.utils import np_utils
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# Create the model
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), activation='relu', padding='same'))
model.add(Dropout(0.2)) # 0.2
model.add(Convolution2D(32, 3, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu', padding='same'))
model.add(Dropout(0.3)) # 0.2
model.add(Convolution2D(64, 3, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, 3, 3, activation='relu', padding='same'))
model.add(Dropout(0.4)) # 0.2
model.add(Convolution2D(128, 3, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', W_constraint=MaxNorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', W_constraint=MaxNorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 30  ##25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)  ### 64
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

print(history.history.keys())
# summarize history for accuracy
_, ax = plt.subplots(2, 1)
ax[0].plot(history.history['acc'],color='b',label='accuracy')
ax[0].plot(history.history['val_acc'],color='r',label='val_accuracy')
ax[0].legend(loc='best',shadow=True)


ax[1].plot(history.history['loss'],color='b',label='loss')
ax[1].plot(history.history['val_loss'],color='r',label='val_loss')
ax[1].legend(loc='best',shadow=True)
plt.show()
