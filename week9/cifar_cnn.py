### HW4 cnn
import keras
import numpy as np
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
from keras import optimizers ### tensorflow 1.14 config
# from tensorflow.keras import optimizers ### tf 2.6 config
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
import imageio
import os
from keras.preprocessing.image import ImageDataGenerator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

K.set_learning_phase(0)

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
		# self.datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, data_format='channels_last')
		# self.datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True,data_format='channels_last')
		# self.datagen = ImageDataGenerator(rotation_range=90, data_format='channels_last')
		# self.datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, data_format='channels_last')
		self.datagen = ImageDataGenerator(zca_whitening=True, data_format='channels_last')

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
			designed_optimizer = optimizers.Adam()
		elif opt=="sgd":
			lrate = 0.01
			decay = lrate / self.num_epochs
			designed_optimizer = optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

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

		self.datagen.fit(self.x_train)

		### if you cannot find the optimal model directly, please use this model to find the best solution
		history = model.fit_generator(self.datagen.flow(self.x_train, self.y_train, batch_size=self.size_batches),
									  steps_per_epoch=len(self.x_train) / self.size_batches,
									  validation_data=(self.x_test, self.y_test), epochs=1)
		model.save("train_cnn_cifar.h5")
		for i in range(self.num_epochs):
			model = keras.models.load_model("train_cnn_cifar.h5")
			history = model.fit_generator(self.datagen.flow(self.x_train, self.y_train, batch_size=self.size_batches),
										  steps_per_epoch=len(self.x_train) / self.size_batches,
										  validation_data=(self.x_test, self.y_test), epochs=1)
			model.save("train_cnn_cifar.h5")
			# if self.acc_max < history.history['val_accuracy'][0]: ### tf 2.6
			if self.acc_max < history.history['val_acc'][0]:  ### tf 1.14
				# self.acc_max = history.history['val_accuracy'][0] ### tf 2.6
				self.acc_max = history.history['val_acc'][0] ### tf 1.14
				self.trained_model = model
		self.trained_model.save("cnn_cifar_zca.h5")

		# history = model.fit_generator(self.datagen.flow(self.x_train, self.y_train, batch_size=self.size_batches),
		# 							  steps_per_epoch=len(self.x_train) / self.size_batches,
		# 							  validation_data=(self.x_test, self.y_test), epochs=self.num_epochs)
		# self.trained_model = model
		#
		#
		# _, ax = plt.subplots(2, 1)
		# ax[0].plot(history.history['acc'], color='b', label='accuracy')  ### tf 1.14
		# ax[0].plot(history.history['val_acc'], color='r', label='val_accuracy')  ### tf 1.14
		# # ax[0].plot(history.history['accuracy'], color='b', label='accuracy') ### tf 2.6
		# # ax[0].plot(history.history['val_accuracy'], color='r', label='val_accuracy') ### tf 2.6
		# ax[0].legend(loc='best', shadow=True)
		# ax[0].grid()
		#
		# ax[1].plot(history.history['loss'], color='b', label='loss')
		# ax[1].plot(history.history['val_loss'], color='r', label='val_loss')
		# ax[1].legend(loc='best', shadow=True)
		# ax[1].grid()
		#
		# plt.show()

	def loadModel(self):

		self.trained_model = keras.models.load_model("cnn_cifar_zca.h5")
		print(self.trained_model.summary())

	def testModel(self):

		self.training_score = self.trained_model.evaluate(self.x_test, self.y_test, verbose=self.verbosity)

	def getScores(self):

		print("CNN Error: %.2f%%" % (100-self.training_score[1]*100))


if __name__ == '__main__':

	epochs, b_size, vbose = 30, 64, 0
	cifar_obj = RecogniseClasses(epochs, b_size, vbose)
	cifar_obj.loadData()
	cifar_obj.prepareData()
	cifar_obj.prepareLabels()
	# created_model = cifar_obj.createModel()
	# cifar_obj.trainModel(created_model)
	cifar_obj.loadModel()
	cifar_obj.testModel()
	cifar_obj.getScores()

	model = cifar_obj.trained_model


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
		input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
		step = 1.
		for i in range(80):
			loss_value, grads_value = iterate([input_img_data])
			input_img_data += grads_value * step

		img = input_img_data[0]
		return deprocess_image(img)

	# ------------------------------------------------------------------------------------------
	# Generating convolution layer filters for intermediate layers using above utility functions
	# ------------------------------------------------------------------------------------------

	# layer_name = 'conv2d_1'
	# preprocess_name = 'zca'
	# size = 32
	# margin = 5
	# results = np.zeros((4 * size + 3 * margin, 8 * size + 7 * margin, 3))
	#
	# for i in range(4):
	# 	for j in range(8):
	# 		filter_img = generate_pattern(layer_name, (i*8) + j, size=size)
	# 		horizontal_start = i * size + i * margin
	# 		horizontal_end = horizontal_start + size
	# 		vertical_start = j * size + j * margin
	# 		vertical_end = vertical_start + size
	# 		results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
	#
	# plt.imshow(results)
	# plt.show()
	#
	# fname = f'filter_layer{layer_name}_{preprocess_name}.jpg'
	#
	# imageio.imwrite(fname,results)
