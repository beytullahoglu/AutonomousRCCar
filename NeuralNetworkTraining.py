# Author: Buğra Beytullahoğlu, https://github.com/beytullahoglu
# I could not be able to finish this project without help of:
# Autonomous RC Car by hamuchiwa, https://github.com/hamuchiwa/AutoRCCar
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# Neural Network Training Code
# Importing necessary packages
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers.convolutional import Conv1D,Conv2D
import keras
import numpy
import tensorflow as tf

# Importing our user input and image files 
X = numpy.loadtxt("imgdata.csv", delimiter = ",")
X = X[1:,:] # taking first row away as it only contains zeros
X = (X*(1/255) - 0.5)*2 # making image array -1 or 1 to avoid any 

Y = numpy.loadtxt("commanddata.csv", delimiter = ",")
Y = Y[1:,:] # taking first row away as it only contains zeros
Y = Y[:,[0,2,3]] # taking second column away as it represents stop command and we do not need it while driving autonomously

Z = Y.argmax(axis=1) # takes the index of command. In that way, analysing output will be easier

# initializing model
model = Sequential()

# Adding layers. To avoid any overfitting, Dropout layers are also be used.
model.add(Dense(750, input_dim = 1536, init = 'he_normal', activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(350, init = 'he_normal', activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(150, init = 'he_normal', activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(3, init = 'he_normal', activation='softmax'))

# Constantly checking validational loss to avoid overfitting
keras.callbacks.ModelCheckpoint(filepath ='callback.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# Compiling our model according to the parameters belove.
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

# Fitting our model finally, 15% of total data will be used to check if there is overfitting.
model.fit(X,Z, epochs=50, batch_size=128, validation_split= 0.15)
predictions = model.predict(X)
predictions = predictions.argmax(axis=1)

# Creating a confussion matrix to see our training results more clearly
from sklearn.metrics import confusion_matrix as cm
conf = cm(Z, predictions, labels=None, sample_weight=None)

# Saving our model in a h5 file
model.save('model.h5')
