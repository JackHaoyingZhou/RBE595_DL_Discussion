### HW4 cnn
import keras
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.constraints import maxnorm
# from keras import optimizers ### tensorflow 1.14 config
from tensorflow.keras import optimizers ### tf 2.6 config
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tf.logging.set_verbosity(tf.logging.ERROR) ### tf 1.14 config
# a = tf.zeros(1) ### tf 1.14 config
#

class RecogniseClasses():

	def __init__(self, epochs, b_size, vbose):
		self.num_classes = None
		self.num_epochs = epochs
		self.size_batches = b_size
		self.verbosity = vbose
		self.acc_max = 0.0

	def loadData(self):
		'''Load the MNIST dataset from Keras'''
		(self.x_train, self.y_train), (self.x_test, self.y_test) =  cifar10.load_data()

	def prepareData(self):
		self.x_train = self.x_train.reshape(self.x_train.shape[0], 32, 32, 3).astype('float32')
		self.x_test	 = self.x_test.reshape(self.x_test.shape[0], 32, 32, 3).astype('float32')
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
			lrate = 0.01
			decay = lrate / self.num_epochs
			designed_optimizer = optimizers.SGD(learning_rate=lrate, momentum=0.9, decay=decay, nesterov=False)

		return designed_optimizer

	def createModel(self):
		model = Sequential()
		model.add(Convolution2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
		model.add(Dropout(0.2))  # 0.2
		model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
		model.add(Dropout(0.3))  # 0.2
		model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
		model.add(Dropout(0.4))  # 0.2
		model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Flatten())
		model.add(Dropout(0.2))
		model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
		model.add(Dropout(0.2))
		model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
		model.add(Dropout(0.2))
		model.add(Dense(self.num_classes, activation='softmax'))
		# Compile model
		opt = self.createOptimizer("sgd")
		model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
		print (model.summary())
		return model

	def trainModel(self, model):

		### if you cannot find the optimal model directly, please use this model to find the best solution
		# model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=1,
		# 					batch_size=self.size_batches)
		# model.save("train_cnn_cifar.h5")
		# for i in range(self.num_epochs):
		# 	model = keras.models.load_model("train_cnn_cifar.h5")
		# 	history = model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=1,
		# 			  batch_size=self.size_batches)
		# 	model.save("train_cnn_cifar.h5")
		# 	if self.acc_max < history.history['val_accuracy'][0]: ### tf 2.6
		#   if self.acc_max < history.history['val_acc'][0]:  ### tf 1.14
		# 		self.acc_max = history.history['val_accuracy'][0] ### tf 2.6
		# 		self.acc_max = history.history['val_acc'][0] ### tf 1.14
		# 		self.trained_model = model
		# self.trained_model.save("cnn_cifar.h5")



		history = model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=self.num_epochs, batch_size=self.size_batches)
		self.trained_model = model
		self.trained_model.save("cnn_cifar.h5")


		_, ax = plt.subplots(2, 1)
		# ax[0].plot(history.history['acc'], color='b', label='accuracy')  ### tf 1.14
		# ax[0].plot(history.history['val_acc'], color='r', label='val_accuracy')  ### tf 1.14
		ax[0].plot(history.history['accuracy'], color='b', label='accuracy') ### tf 2.6
		ax[0].plot(history.history['val_accuracy'], color='r', label='val_accuracy') ### tf 2.6
		ax[0].legend(loc='best', shadow=True)
		ax[0].grid()

		ax[1].plot(history.history['loss'], color='b', label='loss')
		ax[1].plot(history.history['val_loss'], color='r', label='val_loss')
		ax[1].legend(loc='best', shadow=True)
		ax[1].grid()

		plt.show()

	def loadModel(self):

		self.trained_model = keras.models.load_model("cnn_cifar.h5")
		print(self.trained_model.summary())

	def testModel(self):

		self.training_score = self.trained_model.evaluate(self.x_test, self.y_test, verbose=self.verbosity)

	def getScores(self):

		print("CNN Error: %.2f%%" % (100-self.training_score[1]*100))


if __name__ == '__main__':

	epochs, b_size, vbose = 80, 64, 0
	cifar_obj = RecogniseClasses(epochs, b_size, vbose)
	cifar_obj.loadData()
	cifar_obj.prepareData()
	cifar_obj.prepareLabels()
	# created_model = cifar_obj.createModel()
	# cifar_obj.trainModel(created_model)
	cifar_obj.loadModel()
	cifar_obj.testModel()
	cifar_obj.getScores()
