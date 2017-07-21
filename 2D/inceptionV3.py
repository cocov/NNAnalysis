'''
Created on 16 Jun 2017

@author: lyard
'''

#inception V3 model training on Cherenkov data

from keras.applications import InceptionV3
from keras.optimizers import SGD

from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import EarlyStopping

from keras.models import load_model

import numpy as np

from optparse import OptionParser

import h5py

if __name__ == '__main__':
    
    opts_parser = OptionParser()
    opts_parser.add_option("-t", "--training", dest="input", help="training data input hdf5 file", default="", type=str)
    opts_parser.add_option("-v", "--validation", dest="valid", help="validation data input hdf5 file", default="", type=str)
    opts_parser.add_option("-m", "--model", dest="load", help="start from reloaded model filename", default="", type=str)
    opts_parser.add_option("-s", "--save", dest="save", help="save model to target file.", default="", type=str)    
    opts_parser.add_option("-o", "--optimizer", dest="optimizer", help="optimizer to use. default is sgd", default="sgd", type=str)
    opts_parser.add_option("-l", "--loss", dest="loss", help="loss function to use. default is mean_squared_error", default="mean_squared_error", type=str)

    (options, args) = opts_parser.parse_args()
    input_file = options.input
    validation_file = options.valid
    model_file = options.load
    output_file = options.save
    optim_string = options.optimizer
    loss_string  = options.loss
    
    #load input data
    data   = HDF5Matrix(input_file, 'tracesandtiming')
    labels = HDF5Matrix(input_file, 'labels')

    valid_data = HDF5Matrix(validation_file, 'tracesandtiming')
    valid_label = HDF5Matrix(validation_file, 'labels')

    model = InceptionV3(include_top=True, weights=None, input_shape=(299,299,3), pooling='avg', classes=2)

    #learning_rate = 0.01
    #optimizer = SGD(lr=learning_rate, momentum=0.0, decay=0.001, nesterov=False)
    #model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    model.compile(optimizer=optim_string, loss=loss_string, metrics=['accuracy'])

    model.summary()

    if model_file != "":
        model = load_model(model_file)
        print("Reloaded current model from " + model_file)

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=200, verbose=2, mode='min')
    history = model.fit(data, labels, validation_data=[valid_data, valid_label], 
                        epochs=80, batch_size=100, verbose=2, shuffle='batch', callbacks=[early_stop])

    if output_file != "":
        model.save(output_file, True)
        with h5py.File(output_file, 'a') as f:
            if 'optimizer_weights' in f.keys():
                del f['optimizer_weights']

    num_true = 0
    for i in range(valid_data.shape[0]):
        sample = np.reshape(valid_data[i], (1,) + valid_data[i].shape)
        pred  = model.predict(sample)
        if valid_label[i][0] == 0 :
            if pred[0][0] < 0.5 :
                num_true = num_true + 1
                print(str(i) + ": true")
            else:
                print(str(i) + ": FALSE-------")
        else:
            if pred[0][0] >= 0.5 :
                num_true = num_true + 1
                print(str(i) + ": true")
            else:
                print(str(i) + ": FALSE-------")

        
        print("Prediction: [" + str(pred[0][0]) + "," + str(pred[0][1]) + "] vs truth: " + str(labels[i]))
    
    print("Found " + str(num_true*100 / valid_label.shape[0]) + "% of correct samples")
