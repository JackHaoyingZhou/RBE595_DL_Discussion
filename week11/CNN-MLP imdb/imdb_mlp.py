# MLP for the IMDB problem
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load the dataset but only keep the top n words, zero the rest
from sklearn.model_selection import train_test_split
top_words = 5000
test_split = 0.33
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
X_all = numpy.append(X_train,X_test,axis=0)
y_all = numpy.append(y_train,y_test,axis=0)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,test_size=test_split, random_state=0)
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
# create the model
model = Sequential()
model.add(Embedding(top_words, 128, input_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
_, ax = plt.subplots(2, 1)
ax[0].plot(history.history['accuracy'],color='b',label='accuracy')
ax[0].plot(history.history['val_accuracy'],color='r',label='val_accuracy')
ax[0].legend(loc='best',shadow=True)


ax[1].plot(history.history['loss'],color='b',label='loss')
ax[1].plot(history.history['val_loss'],color='r',label='val_loss')
ax[1].legend(loc='best',shadow=True)
plt.show()
