### HW4 mlp
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.layers.convolutional import *
from keras.layers.normalization import BatchNormalization ### tfw 1.14 config
# from tensorflow.keras.layers import BatchNormalization ### tf 2.6 config
from keras.layers import Flatten, Dropout
from keras.metrics import *

from sklearn.metrics import confusion_matrix
import itertools

from keras.utils import np_utils
from keras.optimizers import * ### tfw 1.14 config
# from tensorflow.keras.optimizers import * ### tfw 2.6 config
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR) ### tfw 1.14 config
a = tf.zeros(1) ### tfw 1.14 config

from keras import backend as K
K.set_image_dim_ordering('th') ### tfw 1.14 config

# from keras import backend as K ### tf 2.6 config
# K.set_image_data_format('channels_first') ### tf 2.6 config

class RecogniseHandWrittenDigits():

	def __init__(self, epochs, b_size, vbose):
		self.num_classes = None
		self.num_epochs = epochs
		self.size_batches = b_size
		self.verbosity = vbose

	def loadData(self):
		(self.x_train, self.y_train), (self.x_test, self.y_test) =  mnist.load_data()

	def prepareData(self):
		self.num_pixels = self.x_train.shape[1] * self.x_train.shape[2]
		self.x_train = self.x_train.reshape(self.x_train.shape[0], self.num_pixels).astype('float32')
		self.x_test = self.x_test.reshape(self.x_test.shape[0], self.num_pixels).astype('float32')
		self.x_train, self.x_test = self.x_train/255.0, self.x_test/255.0

	def prepareLabels(self):
		self.y_train = np_utils.to_categorical(self.y_train)
		self.y_test = np_utils.to_categorical(self.y_test)
		self.num_classes = self.y_test.shape[1]

	def createOptimizer(self, opt):

		if opt=="adam":
			designed_optimizer = Adam(learning_rate=0.0001)
		elif opt=="sgd":
			designed_optimizer = SGD(learning_rate=0.1)

		return designed_optimizer

	def createModelMLP(self):
		model = Sequential()

		model.add(Dense(self.num_pixels, input_dim=self.num_pixels, kernel_initializer='normal', activation='relu'))

		model.add(Dense(256, activation="relu"))
		model.add(Dropout(0.2))

		model.add(Dense(self.num_classes, activation="softmax"))

		adm = self.createOptimizer("adam")
		model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=adm)
		print (model.summary())

		return model

	def trainModel(self, model):

		history = model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=self.num_epochs, batch_size=self.size_batches)
		self.trained_model = model
		# self.trained_model.save("mlp_mnist.h5")

		plt.figure()
		plt.subplot(2, 1, 1)
		plt.plot(history.history['acc'], 'b', label="Training Accuracy") ### tfw 1.14 config
		plt.plot(history.history['val_acc'], 'r', label="Validation Accuracy") ### tfw 1.14 config
		# plt.plot(history.history['accuracy'], 'b', label="Training Accuracy") ### tf 2.6 config
		# plt.plot(history.history['val_accuracy'], 'r', label="Validation Accuracy") ### tf 2.6 config
		plt.legend(loc="best")
		plt.grid()
		plt.title('Accuracy')
		plt.subplot(2, 1, 2)
		plt.plot(history.history['loss'], 'b', label="Training Loss")
		plt.plot(history.history['val_loss'], 'r', label="Validation Loss")
		plt.legend(loc="best")
		plt.grid()
		plt.title('Loss')
		plt.show()

	def testModel(self):
		self.training_score = self.trained_model.evaluate(self.x_test, self.y_test, verbose=self.verbosity)

	def printScore(self):
		print("MLP Error: %.2f%%" % (100-self.training_score[1]*100))

if __name__ == '__main__':

	epochs, b_size, vbose = 80, 200, 2
	mnist_obj = RecogniseHandWrittenDigits(epochs, b_size, vbose)
	mnist_obj.loadData()
	mnist_obj.prepareData()
	mnist_obj.prepareLabels()
	created_model = mnist_obj.createModelMLP()
	mnist_obj.trainModel(created_model)
	mnist_obj.testModel()
	mnist_obj.printScore()
