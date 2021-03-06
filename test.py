from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers.recurrent import  LSTM
from keras.layers import Embedding,  Bidirectional
from keras.utils.io_utils import HDF5Matrix
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard,ReduceLROnPlateau
from keras import backend as K

def custom_mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred[:,10:-10] - y_true[:,10:-10]), axis=-1)

def load_data(datapath, train_start, n_training_examples, n_test_examples):
    X_train = HDF5Matrix(datapath, 'traces', train_start, train_start+n_training_examples)
    y_train = HDF5Matrix(datapath, 'labels', train_start, train_start+n_training_examples)
    test_start = train_start+n_training_examples
    X_test = HDF5Matrix(datapath, 'traces', test_start, test_start+n_test_examples)
    y_test = HDF5Matrix(datapath, 'labels', test_start, test_start+n_test_examples)
    return X_train, y_train, X_test, y_test


X_train, Y_train, X_valid, Y_valid = load_data('/data/datasets/CTA/ToyNN/train_0_100_0.hdf5', 0, 9000, 1000)
in_size,out_size = X_train.shape[1],Y_train.shape[1]
print(X_train.shape,Y_train.shape,X_valid.shape,Y_valid.shape)

# expected input data shape: (batch_size, timesteps, data_dim)

model = Sequential()
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(50,return_sequences=True),batch_input_shape=(None,in_size,1),name='bidir_0'))
#model.add(Dropout(0.5))
#model.add(Reshape((out_size,50)))
model.add(Dense(10, activation='relu'))
model.add(Bidirectional(LSTM(10,return_sequences=True),name='bidir_1'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()

#model.compile(loss='mean_squared_error', optimizer='adam')
optimiser = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss=custom_mean_squared_error, optimizer=optimiser)



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
model.fit(X_train, Y_train, nb_epoch=10, batch_size=100, verbose=2,
          validation_data = [X_valid,Y_valid] ,shuffle='batch')
          #,callbacks=[reduce_lr])

#score = model.evaluate(X_valid, Y_valid)
#print(score)
model.save('mymodel.h5')

plt.ion()

plt.figure()
for i in range(100):
    x = X_train[i]
    y = Y_train[i]
    x = x.reshape((1,)+x.shape)
    y = y.reshape((1,)+y.shape)
    pred = model.predict(x)
    y = y.reshape((122,))
    x = x.reshape((122,))
    pred = pred.reshape((122,))
    plt.cla()
    plt.clf()
    plt.step(np.arange(-40,200+4,2),x)
    plt.step(np.arange(-40,200+4,2),y)
    plt.step(np.arange(-40,200+4,2),pred)
    plt.show()
    fk = input('bla')
