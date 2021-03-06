'''
Created on 3 Jul 2017

@author: lyard
'''

from keras.optimizers import Adadelta

from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Input
from keras.layers import AveragePooling2D
from keras import layers

from keras.models import Model

from keras.models import Sequential

from keras.models import load_model

import h5py

from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import EarlyStopping

import numpy as np

from optparse import OptionParser

def normalized_conv2d(x, filters, num_row, num_col, strides=(1,1), padding='same', name=None):
    
    if name is not None:
        bn_name   = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name   = None
        conv_name = None
        
    x = Conv2D(filters, (num_row, num_col), use_bias=False, strides=strides, padding=padding, name=conv_name)(x)
    
    x = BatchNormalization(axis=3, scale=False, name=bn_name)(x)
    
    x = Activation('relu', name=name)(x)
    
    return x
    
def make_mini_inception(input_shape):
     
    image_input = Input(shape=input_shape)                        #56,56,2
    x = normalized_conv2d(image_input, 16, 3, 3, padding='valid') #54,54,32
    x = normalized_conv2d(x,           16, 3, 3, padding='valid') #52,52,32
    x = normalized_conv2d(x,           32, 3, 3)                  #52,52,64
    x = MaxPooling2D((3, 3), strides=(1,1))(x)                    #50,50,64
    
    x = normalized_conv2d(x,           40, 1, 1, padding='valid') #50,50,80
    x = normalized_conv2d(x,           96, 3, 3, padding='valid') #48,48,192
    x = MaxPooling2D((3,3), strides=(2,2))(x)                     #46,46,192
    
    branch1 = normalized_conv2d(x,       32, 1, 1) #5,5,64         46,46,64
    branch5 = normalized_conv2d(x,       24, 1, 1)
    branch5 = normalized_conv2d(branch5, 32, 5, 5) #5,5,64         46,46,64
    branch3 = normalized_conv2d(x,       32, 1, 1)
    branch3 = normalized_conv2d(branch3, 48, 3, 3)
    branch3 = normalized_conv2d(branch3, 48, 3, 3) #5,5,96         46,46,96
    
    branchp = AveragePooling2D((3,3), strides=(1,1), padding='same')(x)
    branchp = normalized_conv2d(branchp, 32, 1, 1) 
    
    x = layers.concatenate([branch1, branch5, branch3, branchp], axis=3, name='mixed0')
    
    branch1 = normalized_conv2d(x, 32, 1, 1)
    branch5 = normalized_conv2d(x, 24, 1, 1)
    branch5 = normalized_conv2d(branch5, 32, 5, 5)
    branch3 = normalized_conv2d(x, 32, 1, 1)
    branch3 = normalized_conv2d(branch3, 48, 3, 3)
    branch3 = normalized_conv2d(branch3, 48, 3, 3)
    branchp = AveragePooling2D((3,3), strides=(1,1), padding='same')(x)
    branchp = normalized_conv2d(branchp, 32, 1, 1)
    x = layers.concatenate([branch1, branch5, branch3, branchp], axis=3, name='mixed1')
    
    branch1 = normalized_conv2d(x, 32, 1, 1)
    branch5 = normalized_conv2d(x, 24, 1, 1)
    branch5 = normalized_conv2d(branch5, 32, 5, 5)
    branch3 = normalized_conv2d(x, 32, 1, 1)
    branch3 = normalized_conv2d(branch3, 48, 3, 3)
    branch3 = normalized_conv2d(branch3, 48, 3, 3)
    branchp = AveragePooling2D((3,3), strides=(1,1), padding='same')(x)
    branchp = normalized_conv2d(branchp, 32, 1, 1)
    x = layers.concatenate([branch1, branch5, branch3, branchp], axis=3, name='mixed2')
    
    branch3 = normalized_conv2d(x, 192, 3, 3, strides=(2,2), padding='valid')
    branch33 = normalized_conv2d(x, 32, 1, 1)
    branch33 = normalized_conv2d(branch33, 48, 3, 3)
    branch33 = normalized_conv2d(branch33, 48, 3, 3, strides=(2,2), padding='valid')
    branchp = MaxPooling2D((3,3), strides=(2,2))(x)
    x = layers.concatenate([branch3, branch33, branchp], axis=3, name='mixed3')
    
    branch1 = normalized_conv2d(x,  96, 1, 1)
    branch7 = normalized_conv2d(x,  64, 1, 1)
    branch7 = normalized_conv2d(branch7, 64, 1, 5)
    branch7 = normalized_conv2d(branch7, 96, 7, 1)
    branch77 = normalized_conv2d(x, 64, 1, 1)
    branch77 = normalized_conv2d(branch77, 64, 5, 1)
    branch77 = normalized_conv2d(branch77, 64, 1, 5)
    branch77 = normalized_conv2d(branch77, 64, 5, 1)
    branch77 = normalized_conv2d(branch77, 96, 1, 5)
    branchp = AveragePooling2D((3,3), strides=(1,1), padding='same')(x)
    branchp = normalized_conv2d(branchp, 96, 1, 1)
    x = layers.concatenate([branch1, branch7, branch77, branchp], axis=3, name='mixed4')
     
    for i in range(2):
        branch1 = normalized_conv2d(x, 96, 1, 1)
        branch7 = normalized_conv2d(x, 80, 1, 1)
        branch7 = normalized_conv2d(branch7, 80, 1, 5)
        branch7 = normalized_conv2d(branch7, 96, 5, 1)
        branch77 = normalized_conv2d(x, 80, 1, 1)
        branch77 = normalized_conv2d(branch77, 80, 5, 1)
        branch77 = normalized_conv2d(branch77, 80, 1, 5)
        branch77 = normalized_conv2d(branch77, 80, 5, 1)
        branch77 = normalized_conv2d(branch77, 96, 1, 5)
        branchp = AveragePooling2D((3,3), strides=(1,1), padding='same')(x)
        branchp = normalized_conv2d(branchp, 96, 1, 1)
        x = layers.concatenate([branch1, branch7, branch77, branchp], axis=3, name='mixed'+str(5+i))
        
    branch1 = normalized_conv2d(x, 96, 1, 1)
    branch7 = normalized_conv2d(x, 96, 1, 1)
    branch7 = normalized_conv2d(branch7, 96, 1, 5)
    branch7 = normalized_conv2d(branch7, 96, 5, 1)
    branch77 = normalized_conv2d(x, 96, 1, 1)
    branch77 = normalized_conv2d(branch77, 96, 5, 1)
    branch77 = normalized_conv2d(branch77, 96, 1, 5)
    branch77 = normalized_conv2d(branch77, 96, 5, 1)
    branch77 = normalized_conv2d(branch77, 96, 1, 5)
    branchp = AveragePooling2D((3,3), strides=(1,1), padding='same')(x)
    branchp = normalized_conv2d(branchp, 96, 1, 1)
    x = layers.concatenate([branch1, branch7, branch77, branchp], axis=3, name='mixed7')
    
    branch3 = normalized_conv2d(x, 96, 1, 1)
    branch3 = normalized_conv2d(branch3, 160, 3, 3, strides=(2,2), padding='valid')
    branch7 = normalized_conv2d(x, 96, 1, 1)
    branch7 = normalized_conv2d(branch7, 96, 1, 5)
    branch7 = normalized_conv2d(branch7, 96, 5, 1)
    branch7 = normalized_conv2d(branch7, 96, 3, 3, strides=(2,2), padding='valid')
    branchp = MaxPooling2D((3,3), strides=(2,2))(x)
    x = layers.concatenate([branch3, branch7, branchp], axis=3, name='mixed8')
    
    for i in range(2):
        branch1 = normalized_conv2d(x, 160, 1, 1)
        branch3 = normalized_conv2d(x, 192, 1, 1)
        branch31 = normalized_conv2d(branch3, 192, 1, 3)
        branch32 = normalized_conv2d(branch3, 192, 3, 1)
        branch3 = layers.concatenate([branch31, branch32], axis=3, name='mixed9'+str(i))
        
        branch3dbl = normalized_conv2d(x, 224, 1, 1)
        branch3dbl = normalized_conv2d(branch3dbl, 192, 3, 3)
        branch3dbl1 = normalized_conv2d(branch3dbl, 192, 1, 3)
        branch3dbl2 = normalized_conv2d(branch3dbl, 192, 3, 1)
        branch3dbl = layers.concatenate([branch3dbl1, branch3dbl2], axis=3)
        
        branchp = AveragePooling2D((3,3), strides=(1,1), padding='same')(x)
        branchp = normalized_conv2d(branchp, 96, 1, 1)
        x = layers.concatenate([branch1, branch3, branch3dbl, branchp], axis=3, name='mixed'+str(9+i))
        
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(2, activation='softmax', name='predictions')(x)
    
    model = Model(image_input, x, name='miniception')
    
    return model
    
    
if __name__ == '__main__':
    
    opts_parser = OptionParser()
    opts_parser.add_option("-t", "--training", dest="input", help="training dataset", default="", type=str)
    opts_parser.add_option("-v", "--validation", dest="valid", help="validation dataset", default="", type=str)
    opts_parser.add_option("-m", "--model", dest="load", help="start from reloaded model", default="", type=str)
    opts_parser.add_option("-s", "--save", dest="save", help="save trained model to target file", default="", type=str)
 
    (options, args) = opts_parser.parse_args()
    
    training_file   = options.input
    validation_file = options.valid
    
    model_file  = options.load
    target_file = options.save

    data   = HDF5Matrix(training_file, 'values')
    labels = HDF5Matrix(training_file, 'labels')
    
    valid_data = HDF5Matrix(validation_file, 'values')
    valid_labels = HDF5Matrix(validation_file, 'labels')

    model = make_mini_inception(input_shape=(56, 56, 1))
 
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    if model_file != "":
        model = load_model(model_file)
        print("Reloaded model from " + model_file)

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=100, verbose=2, mode='min')
    history = model.fit(data, labels, validation_data=[valid_data, valid_labels], 
                        epochs=80, batch_size=100, verbose=2, shuffle='batch', callbacks=[early_stop])

    if target_file != "":
        model.save(target_file, True)
        print("wrote model to " + target_file)
        with h5py.File(target_file, 'a') as f:
            if 'optimizer_weights' in f.keys():
                del f['optimizer_weights']

    num_true = 0
    for i in range(valid_data.shape[0]):
        sample = np.reshape(valid_data[i], (1,) + valid_data[i].shape)
        prediction = model.predict(sample)
        print("Prediction: [" + str(prediction[0][0]) + "," + str(prediction[0][1]) + "] vs truth: " + str(valid_labels[i]))


        
