import math
import numpy as np

from utility import visualizeOutput

from keras.models import Sequential
from keras.layers.core import Dense
from keras.datasets import cifar10

from keras.layers.convolutional import *
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import *

from keras import backend as K

from sklearn.metrics import confusion_matrix
import itertools

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from keras.utils import np_utils
from tensorflow.keras import optimizers

from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16

import time
from keras.preprocessing.image import save_img

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import backend as K
K.set_image_data_format('channels_first')

class RecognizeClasses():

	def __init__(self, epochs, b_size, vbose):
		self.num_classes = None
		self.num_epochs = epochs
		self.size_batches = b_size
		self.verbosity = vbose

	def loadData(self):
		(self.x_train, self.y_train), (self.x_test, self.y_test) =  cifar10.load_data()

	def prepareData(self):
		self.num_pixels = self.x_train.shape[1] * self.x_train.shape[2]
		self.x_train = self.x_train.reshape(self.x_train.shape[0], self.num_pixels).astype('float32')
		self.x_test = self.x_test.reshape(self.x_test.shape[0], self.num_pixels).astype('float32')
		self.x_train, self.x_test = self.x_train/255.0, self.x_test/255.0

	def prepareLabels(self):
		self.y_train = np_utils.to_categorical(self.y_train)
		self.y_test = np_utils.to_categorical(self.y_test)
		self.num_classes = self.y_test.shape[1]

	def defineVisFilterDims(self):
		self.img_height, self.img_width = 128,128

	def createOptimizer(self, opt):

		lr = 0.1
		dcy = lr/self.num_epochs
		if opt=="adam":
			designed_optimizer = optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, decay=dcy)
			designed_optimizer = optimizers.Adam()
		elif opt=="sgd":
			designed_optimizer = optimizers.SGD(learning_rate=0.1)

		return designed_optimizer

	def obtainTrainedModel(self):
		# model = MobileNet(input_shape=(224,224,3), alpha=1.0, dropout=1e-3, include_top=False, weights='imagenet', input_tensor=None, classes=10)
		model = Sequential()

		model.add(Convolution2D(32, (5, 5), activation="relu", input_shape=(1, 28, 28), padding="valid"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))

		model.add(Flatten())

		model.add(Dense(128, activation="sigmoid"))

		# model.add(Dense(32, activation="relu"))
		model.add(Dense(self.num_classes, activation="softmax"))

		adm = self.createOptimizer("adam")
		model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=adm)
		# print(model.summary())

		self.trained_model = model

		for idx in range(len(model.layers)):
			print(model.get_layer(index=idx).name)

	def trainModel(self, model):

		model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=self.num_epochs, batch_size=self.size_batches)
		# self.trained_model = model
		# # self.trained_model.save("cnn_mnist.h5")
		# plt.figure()
		# plt.subplot(2,1,1)
		# plt.plot(history.history['acc'],'b',label="Training Accuracy") ### tensorflow 1.14 config
		# plt.plot(history.history['val_acc'], 'r', label="Validation Accuracy") ### tensorflow 1.14 config
		# # plt.plot(history.history['accuracy'], 'b', label="Training Accuracy") ### tf 2.6 config
		# # plt.plot(history.history['val_accuracy'], 'r', label="Validation Accuracy") ### tf 2.6 config
		# plt.legend(loc="best")
		# plt.grid()
		# plt.title('Accuracy')
		# plt.subplot(2, 1, 2)
		# plt.plot(history.history['loss'],'b',label="Training Loss")
		# plt.plot(history.history['val_loss'], 'r', label="Validation Loss")
		# plt.legend(loc="best")
		# plt.grid()
		# plt.title('Loss')
		# plt.show()

	def getModelSummary(self):
		print (self.trained_model.summary())

	def testModel(self):
		self.training_score = self.trained_model.evaluate(self.x_test, self.y_test, verbose=self.verbosity)

	def getNumLayers(self):
		print("Number of layers: %s " %(len(self.trained_model.layers)))

	def visualizeFilter(self, layer_id):
		# Visualize weights
		self.layer_id =layer_id
		self.layer_to_see = self.trained_model.layers[layer_id]
		self.layer_to_see_name = 'conv1'

		print("Image data shape- ", self.layer_to_see.get_weights()[0][:, :, :, 0].squeeze().shape)
		print("Image data type- ", type(self.layer_to_see.get_weights()[0][:, :, :, 0].squeeze()))
		if (self.layer_to_see.get_weights()!= []):
			all_images = self.layer_to_see.get_weights()[0][:, :, :, 0].squeeze()
			print("Shape --- ", all_images.shape)
			(m, n, r) = all_images.shape
			all_images_concatenated = all_images.reshape( int(math.ceil(m*math.sqrt(r))), int(math.ceil(m*math.sqrt(r))))
			imgplot = plt.imshow(all_images_concatenated)
			plt.show()

	def visualizeLayer(self, layer_name):
		visualizeOutput(self.trained_model, l_name=layer_name, num_filters=32)

	def printScore(self):
		print("CNN Error: %.2f%%" % (100-self.training_score[1]*100))

	def processLayer(self):
		# this is the placeholder for the input images
		self.input_img = self.trained_model.input

		# get the symbolic outputs of each "key" layer (we gave them unique names).
		self.layer_dict = dict([(layer.name, layer) for layer in self.trained_model.layers[1:]])

	def plotFilter(self,layer_name='block1_conv2'):
		kept_filters = []
		for filter_index in range(200):
		    # we only scan through the first 200 filters,
		    # but there are actually 512 of them
		    print('Processing filter %d' % filter_index)
		    start_time = time.time()

		    # we build a loss function that maximizes the activation
		    # of the nth filter of the layer considered
		    layer_output = self.layer_dict[layer_name].output
		    if K.image_data_format() == 'channels_first':
		        loss = K.mean(layer_output[:, filter_index, :, :])
		    else:
		        loss = K.mean(layer_output[:, :, :, filter_index])

		    # we compute the gradient of the input picture wrt this loss
		    grads = K.gradients(loss, self.input_img)[0]

		    # normalization trick: we normalize the gradient
		    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5) #grads = self.normalize(grads)


		    # this function returns the loss and grads given the input picture
		    iterate = K.function([self.input_img], [loss, grads])

		    # step size for gradient ascent
		    step = 1.

		    # we start from a gray image with some random noise
		    input_img_data = (self.input_img - 0.5) * 20 + 128

		    # we run gradient ascent for 20 steps
		    for i in range(20):
		        loss_value, grads_value = iterate([input_img_data])
		        input_img_data += grads_value * step

		        print('Current loss value:', loss_value)
		        if loss_value <= 0.:
		            # some filters get stuck to 0, we can skip them
		            break

		    # decode the resulting input image
		    if loss_value > 0:
		        img = self.deprocess_image(input_img_data[0])
		        kept_filters.append((img, loss_value))
		    end_time = time.time()
		    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

		# we will stich the best 64 filters on a 8 x 8 grid.
		n = 8

		# the filters that have the highest loss are assumed to be better-looking.
		# we will only keep the top 64 filters.
		kept_filters.sort(key=lambda x: x[1], reverse=True)
		kept_filters = kept_filters[:n * n]

		# build a black picture with enough space for
		# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
		margin = 5
		width = n * img_width + (n - 1) * margin
		height = n * img_height + (n - 1) * margin
		stitched_filters = np.zeros((width, height, 3))

		# fill the picture with our saved filters
		for i in range(n):
		    for j in range(n):
		        img, loss = kept_filters[i * n + j]
		        width_margin = (img_width + margin) * i
		        height_margin = (img_height + margin) * j
		        stitched_filters[
		            width_margin: width_margin + img_width,
		            height_margin: height_margin + img_height, :] = img

	def execute(self):
		self.loadData()
		self.prepareData()
		self.prepareLabels()
		self.obtainTrainedModel()
		self.getModelSummary()
		self.getNumLayers()
		# self.processLayer()
		# self.plotFilter()
		# self.visualizeLayer('block5_conv3')


if __name__ == '__main__':

	epochs, b_size, vbose = 50, 200, 2
	cifar10_obj = RecognizeClasses(epochs, b_size, vbose)
	cifar10_obj.execute()
