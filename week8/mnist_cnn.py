### HW4 cnn

from utility import visualizeOutput
import keras
import numpy as np
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
import imageio
from keras import backend as K
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

K.set_learning_phase(0)

# tf.logging.set_verbosity(tf.logging.ERROR) ### tf 1.14 config
# a = tf.zeros(1) ### tf 1.14 config
#

class RecogniseHandWrittenDigits():

	def __init__(self, epochs, b_size, vbose):
		self.num_classes = None
		self.num_epochs = epochs
		self.size_batches = b_size
		self.verbosity = vbose
		self.acc_max = 0.0

	def loadData(self):
		'''Load the MNIST dataset from Keras'''
		(self.x_train, self.y_train), (self.x_test, self.y_test) =  mnist.load_data()

	def prepareData(self):
		self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1).astype('float32')
		self.x_test	 = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1).astype('float32')
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

		model.add(Convolution2D(32, (5,5), activation="relu", input_shape=(28, 28, 1), padding="valid"))
		model.add(Convolution2D(32, (5, 5), activation="relu", padding="same"))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))

		model.add(Convolution2D(64, (3,3), activation="relu", input_shape=(28, 28, 1), padding="same"))
		model.add(Convolution2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1), padding="same"))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
		model.add(Dropout(0.25))

		model.add(Flatten())

		model.add(Dense(256, activation="relu"))
		model.add(Dropout(0.25))
		model.add(Dense(self.num_classes, activation="softmax"))

		adm = self.createOptimizer("adam")
		model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=adm)
		print (model.summary())
		return model

	def trainModel(self, model):

		### if you cannot find the optimal model directly, please use this model to find the best solution
		# model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=1,
		# 					batch_size=self.size_batches)
		# model.save("train_cnn_mnist.h5")
		# for i in range(self.num_epochs):
		# 	model = keras.models.load_model("train_cnn_mnist.h5")
		# 	history = model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=1,
		# 			  batch_size=self.size_batches)
		# 	model.save("train_cnn_mnist.h5")
		# 	if self.acc_max < history.history['val_accuracy'][0]: ### tf 2.6
		#   if self.acc_max < history.history['val_acc'][0]:  ### tf 1.14
		# 		self.acc_max = history.history['val_accuracy'][0] ### tf 2.6
		# 		self.acc_max = history.history['val_acc'][0] ### tf 1.14
		# 		self.trained_model = model
		# self.trained_model.save("cnn_mnist.h5")



		history = model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=self.num_epochs, batch_size=self.size_batches)
		self.trained_model = model


		_, ax = plt.subplots(2, 1)
		ax[0].plot(history.history['acc'], color='b', label='accuracy')  ### tf 1.14
		ax[0].plot(history.history['val_acc'], color='r', label='val_accuracy')  ### tf 1.14
		# ax[0].plot(history.history['accuracy'], color='b', label='accuracy') ### tf 2.6
		# ax[0].plot(history.history['val_accuracy'], color='r', label='val_accuracy') ### tf 2.6
		ax[0].legend(loc='best', shadow=True)
		ax[0].grid()

		ax[1].plot(history.history['loss'], color='b', label='loss')
		ax[1].plot(history.history['val_loss'], color='r', label='val_loss')
		ax[1].legend(loc='best', shadow=True)
		ax[1].grid()

		plt.show()

	def visualizeLayer(self, layer_name):
		visualizeOutput(self.trained_model, l_name=layer_name, num_filters=32)

	def loadModel(self):

		self.trained_model = keras.models.load_model("cnn_mnist_tf1.h5")
		print(self.trained_model.summary())

	def testModel(self):

		self.training_score = self.trained_model.evaluate(self.x_test, self.y_test, verbose=self.verbosity)

	def getScores(self):

		print("CNN Error: %.2f%%" % (100-self.training_score[1]*100))


if __name__ == '__main__':

	epochs, b_size, vbose = 25, 128, 0
	mnist_obj = RecogniseHandWrittenDigits(epochs, b_size, vbose)
	mnist_obj.loadData()
	mnist_obj.prepareData()
	mnist_obj.prepareLabels()
	# created_model = mnist_obj.createModel()
	# mnist_obj.trainModel(created_model)
	mnist_obj.loadModel()
	# mnist_obj.visualizeLayer('conv2d')
	mnist_obj.testModel()
	mnist_obj.getScores()

	model = mnist_obj.trained_model

	def deprocess_image(x):

		x -= x.mean()
		x /= (x.std() + 1e-5)
		x *= 0.1
		x += 0.5
		x = np.clip(x, 0, 1)
		x *= 255
		x = np.clip(x, 0, 255).astype('uint8')
		return x


	# ---------------------------------------------------------------------------------------------------
	# Utility function for generating patterns for given layer starting from empty input image and then
	# applying Stochastic Gradient Ascent for maximizing the response of particular filter in given layer
	# ---------------------------------------------------------------------------------------------------

	def generate_pattern(layer_name, filter_index, size=150):

		layer_output = model.get_layer(layer_name).output
		loss = K.mean(layer_output[:, :, :, filter_index])
		grads = K.gradients(loss, model.input)[0]
		grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
		iterate = K.function([model.input], [loss, grads])
		input_img_data = np.random.random((1, size, size, 1)) * 20 + 128.
		step = 1.
		for i in range(80):
			loss_value, grads_value = iterate([input_img_data])
			input_img_data += grads_value * step

		img = input_img_data[0]
		return deprocess_image(img)


	# def save_img(self, img, fname):
	# 	pil_img = deprocess_image(np.copy(img))
	# 	imageio.imwrite(fname, pil_img)

	# ------------------------------------------------------------------------------------------
	# Generating convolution layer filters for intermediate layers using above utility functions
	# ------------------------------------------------------------------------------------------

	layer_name = 'conv2d_1'
	size = 28
	margin = 5
	results = np.zeros((8 * size + 7 * margin, 4 * size + 3 * margin, 1))

	for i in range(8):
		for j in range(4):
			filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
			horizontal_start = i * size + i * margin
			horizontal_end = horizontal_start + size
			vertical_start = j * size + j * margin
			vertical_end = vertical_start + size
			results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

	plt.imshow(results)
	plt.show()

	fname = f'filter_layer{layer_name}.jpg'

	imageio.imwrite(fname,results)
