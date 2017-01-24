
from keras.utils.io_utils import HDF5Matrix
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
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


X_train, Y_train, X_valid, Y_valid = load_data('/data/datasets/CTA/ToyNN/train_40_200_0.hdf5', 0, 18000, 2000)
in_size,out_size = X_train.shape[1],Y_train.shape[1]
print(X_train.shape,Y_train.shape,X_valid.shape,Y_valid.shape)

model = load_model('model_test.h5', custom_objects={'custom_mean_squared_error': custom_mean_squared_error})
model.summary()



plt.ion()

plt.figure()
for i in range(100):
    x = X_train[i]
    y = Y_train[i]
    x = x.reshape((1,)+x.shape)
    y = y.reshape((1,)+y.shape)
    pred = model.predict(x)
    y = y.reshape((152,))
    x = x.reshape((152,))
    pred = pred.reshape((152,))
    plt.cla()
    plt.clf()
    plt.step(np.arange(-150,150+4,2),x)
    plt.step(np.arange(-150,150+4,2),y)
    plt.step(np.arange(-150,150+4,2),pred)
    plt.show()
    fk = input('bla')
