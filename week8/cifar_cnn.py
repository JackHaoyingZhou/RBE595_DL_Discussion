### HW4 cnn
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import optimizers ### tensorflow 1.14 config
# from tensorflow.keras import optimizers ### tf 2.6 config
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR) ### tf 1.14 config
a = tf.zeros(1) ### tf 1.14 config

from keras import backend as K ### tf 1.14 config
K.set_image_dim_ordering('th') ### tf 1.14 config

# from keras import backend as K ### tf 2.6 config
# K.set_image_data_format('channels_first') ### tf 2.6 config

class RecogniseHandWrittenDigits():

	def __init__(self, epochs, b_size, vbose):
		self.num_classes = None
		self.num_epochs = epochs
		self.size_batches = b_size
		self.verbosity = vbose

	def loadData(self):
		'''Load the MNIST dataset from Keras'''
		(self.x_train, self.y_train), (self.x_test, self.y_test) =  mnist.load_data()

	def prepareData(self):
		self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, 28, 28).astype('float32')
		self.x_test	 = self.x_test.reshape(self.x_test.shape[0], 1, 28, 28).astype('float32')
		self.x_train, self.x_test = self.x_train/255.0, self.x_test/255.0

	def prepareLabels(self):
		self.y_train = np_utils.to_categorical(self.y_train)
		self.y_test = np_utils.to_categorical(self.y_test)
		self.num_classes = self.y_test.shape[1]

	def createOptimizer(self, opt):

		lr = 0.1
		dcy = lr/self.num_epochs
		if opt=="adam":
			designed_optimizer = optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, decay=dcy)
			designed_optimizer = optimizers.Adam()
		elif opt=="sgd":
			designed_optimizer = optimizers.SGD(learning_rate=0.1)

		return designed_optimizer

	def createModel(self):
		model = Sequential()

		model.add(Convolution2D(32, (5,5), activation="relu", input_shape=(1, 28, 28), padding="valid"))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.2))

		model.add(Flatten())

		model.add(Dense(128, activation="sigmoid"))

		# model.add(Dense(32, activation="relu"))
		model.add(Dense(self.num_classes, activation="softmax"))

		adm = self.createOptimizer("adam")
		model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=adm)
		print (model.summary())
		return model

	def trainModel(self, model):

		history = model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=self.num_epochs, batch_size=self.size_batches)
		self.trained_model = model
		# self.trained_model.save("cnn_mnist.h5")
		plt.figure()
		plt.subplot(2,1,1)
		plt.plot(history.history['acc'],'b',label="Training Accuracy") ### tensorflow 1.14 config
		plt.plot(history.history['val_acc'], 'r', label="Validation Accuracy") ### tensorflow 1.14 config
		# plt.plot(history.history['accuracy'], 'b', label="Training Accuracy") ### tf 2.6 config
		# plt.plot(history.history['val_accuracy'], 'r', label="Validation Accuracy") ### tf 2.6 config
		plt.legend(loc="best")
		plt.grid()
		plt.title('Accuracy')
		plt.subplot(2, 1, 2)
		plt.plot(history.history['loss'],'b',label="Training Loss")
		plt.plot(history.history['val_loss'], 'r', label="Validation Loss")
		plt.legend(loc="best")
		plt.grid()
		plt.title('Loss')
		plt.show()

	def testModel(self):

		self.training_score = self.trained_model.evaluate(self.x_test, self.y_test, verbose=self.verbosity)

	def getScores(self):

		print("CNN Error: %.2f%%" % (100-self.training_score[1]*100))


if __name__ == '__main__':

	epochs, b_size, vbose = 20, 200, 0
	mnist_obj = RecogniseHandWrittenDigits(epochs, b_size, vbose)
	mnist_obj.loadData()
	mnist_obj.prepareData()
	mnist_obj.prepareLabels()
	created_model = mnist_obj.createModel()
	mnist_obj.trainModel(created_model)
	mnist_obj.testModel()
	mnist_obj.getScores()
