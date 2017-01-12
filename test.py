
import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import  LSTM
from keras.layers import Embedding,  Bidirectional
from keras.utils.io_utils import HDF5Matrix
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard,ReduceLROnPlateau
from keras import backend as K

def custom_mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred[:,5:45] - y_true[:,5:45]), axis=-1)

def load_data(datapath, train_start, n_training_examples, n_test_examples):
    X_train = HDF5Matrix(datapath, 'traces', train_start, train_start+n_training_examples)
    y_train = HDF5Matrix(datapath, 'labels', train_start, train_start+n_training_examples)
    test_start = train_start+n_training_examples
    X_test = HDF5Matrix(datapath, 'traces', test_start, test_start+n_test_examples)
    y_test = HDF5Matrix(datapath, 'labels', test_start, test_start+n_test_examples)
    return X_train, y_train, X_test, y_test


# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(Bidirectional(LSTM(51,return_sequences=True),batch_input_shape=(None,51,1)))  # returns a sequence of vectors of dimension 32
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(51,return_sequences=True)))  # returns a sequence of vectors of dimension 32
model.add(Dropout(0.5))
#model.add(Dense(51))
#model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))

'''
model.add(LSTM(51,return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("linear"))
'''
#model.compile(loss='mean_squared_error', optimizer='adam')
optimiser = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss=custom_mean_squared_error, optimizer=optimiser)


X_train, Y_train,X_valid,Y_valid = load_data('/data/datasets/CTA/ToyNN/train_0_660_0.hdf5', 0, 90000, 10000)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.0001)
model.fit(X_train, Y_train, nb_epoch=20, batch_size=100, verbose=2,
          validation_data = [X_valid,Y_valid] ,shuffle='batch',
          callbacks=[reduce_lr])

#score = model.evaluate(X_valid, Y_valid)
#print(score)
model.save('mymodel.h5')
from keras.utils.visualize_util import plot
plot(model, to_file='model.png')

plt.ion()

plt.figure()
for i in range(100):
    x = X_train[i]
    y = Y_train[i]
    x = x.reshape((1,)+x.shape)
    y = y.reshape((1,)+y.shape)
    pred = model.predict(x)
    y = y.reshape((51,))
    x = x.reshape((51,))
    pred = pred.reshape((51,))
    plt.cla()
    plt.clf()
    plt.step(np.arange(10,40,1),x[10:40:1])
    plt.step(np.arange(10,40,1),y[10:40:1])
    plt.step(np.arange(10,40,1),pred[10:40:1])
    plt.show()
    fk = input('bla')
