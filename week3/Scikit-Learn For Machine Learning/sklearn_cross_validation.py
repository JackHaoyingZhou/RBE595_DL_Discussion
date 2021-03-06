# Carlos Morato, PhD.
# cwmorato@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
import numpy

# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
	model.add(Dense(8, init='uniform', activation='relu'))
	model.add(Dense(1, init='uniform', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, nb_epoch=150, batch_size=10, verbose=0)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(y=Y, n_folds=10, shuffle=True, random_state=seed)
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# kfold = skf.split(X,Y)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
