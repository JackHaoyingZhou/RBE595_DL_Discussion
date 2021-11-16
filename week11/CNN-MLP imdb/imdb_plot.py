# Load and Plot the IMDB dataset
import numpy
from keras.datasets import imdb
from matplotlib import pyplot
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# load the dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data()
X_train = numpy.append(X_train,X_test,axis=0)
y_train = numpy.append(y_train,y_test,axis=0)
# summarize size
print("Training data: ")
print(X_train.shape)
print(y_train.shape)
# Summarize number of classes
print("Classes: ")
print(numpy.unique(y_train))
# Summarize number of words
print("Number of words: ")
print(len(numpy.unique(numpy.hstack(X_train))))
# Summarize review length
print("Review length: ")
result = list(map(len, X_train))
print("Mean %.2f words (%f)" % (numpy.mean(result), numpy.std(result)))
# plot review length as a boxplot and histogram
pyplot.subplot(121)
pyplot.boxplot(result)
pyplot.subplot(122)
pyplot.hist(result)
pyplot.show()
