# Carlos Morato, PhD.
# cwmorato@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# LSTM and CNN for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from theano.tensor.shared_randomstreams import RandomStreams
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# fix random seed for reproducibility
numpy.random.seed(7)
srng = RandomStreams(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
test_split = 0.33
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
X_all = numpy.append(X_train,X_test,axis=0)
y_all = numpy.append(y_train,y_test,axis=0)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,test_size=test_split, random_state=0)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32 #32 original
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Convolution1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
# print(history.history.keys())

_, ax = plt.subplots(2, 1)
ax[0].plot(history.history['accuracy'],color='b',label='accuracy')
ax[0].plot(history.history['val_accuracy'],color='r',label='val_accuracy')
ax[0].legend(loc='best',shadow=True)


ax[1].plot(history.history['loss'],color='b',label='loss')
ax[1].plot(history.history['val_loss'],color='r',label='val_loss')
ax[1].legend(loc='best',shadow=True)
plt.show()
