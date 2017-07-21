'''
Created on 13 Jun 2017

@author: lyard
'''

#from data_import import squarization
from optparse import OptionParser

from keras.utils.io_utils import HDF5Matrix

#from ctapipe.instrument import CameraGeometry
#from ctapipe.visualization import CameraDisplay
#needed to create our own camera geometry for visualization
#from astropy import units as u

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import load_model
from keras.callbacks import EarlyStopping

from keras.optimizers import SGD

import keras

#import matplotlib.pyplot as plot
#from ctapipe.image.tests.test_geometry_converter import plot_cam

if __name__ == '__main__':
    
    opts_parser = OptionParser()
    opts_parser.add_option("-t", "--training", dest="input", help="training data input hdf5 file", default="", type=str)
    opts_parser.add_option("-v", "--validation", dest="valid", help="validation data input hdf5 file", default="", type=str)
    opts_parser.add_option("-l", "--load", dest="load", help="start from reloaded model filename", default="", type=str)
    
    (options, args) = opts_parser.parse_args()
    input_file = options.input
    validation_file = options.valid
    model_file = options.load
    
    #load input data
    data   = HDF5Matrix(input_file, 'traces')
    labels = HDF5Matrix(input_file, 'labels')
    
    valid_data = HDF5Matrix(validation_file, 'traces')
    valid_label = HDF5Matrix(validation_file, 'labels')
    
    #display the first event just to be sure 
#    cam_geom = CameraGeometry.from_name("FlashCam")
#    sqr_geom = squarization.build_square_geometry(cam_geom)
        
    #create a square geometry
    matrix_size       = 56
#    square_num_pixels = matrix_size * matrix_size
#    square_pix_ids = np.arange(square_num_pixels)
#    square_x_pos   = np.empty(square_num_pixels).tolist()
#    square_y_pos   = np.empty(square_num_pixels).tolist()
#    pixel_size     = 0.01 #meters
#    pixel_area     = np.empty(square_num_pixels).tolist()#pixel_size * pixel_size
    
#    for i in range(0, matrix_size) :
#        for j in range(0, matrix_size) :
#            square_x_pos[i*matrix_size + j] = i*pixel_size
#            square_y_pos[i*matrix_size + j] = j*pixel_size
#            pixel_area[i*matrix_size + j] = pixel_size * pixel_size
            
#    square_camera = CameraGeometry(cam_id   = "SquashCam", 
#                                   pix_id   = square_pix_ids, 
#                                   pix_x    = square_x_pos * u.meter, 
#                                   pix_y    = square_y_pos * u.meter, 
#                                   pix_area = pixel_area, 
#                                   pix_type = 'rectangular')

#    disp = CameraDisplay(square_camera)
#    disp.add_colorbar()
    
#    disp.image = data[0]
#    disp.show()
#    stop = input('Hit a key to start learning from this data')
    
    #now start doing the neural networks stuff
    #add a new axis to an array (for convolutions): x[:,newaxis]
    
    print(data.shape)
    data2 = np.expand_dims(data, axis=3)
    valid_data2 = np.expand_dims(valid_data, axis=3)
    print(data2.shape)

    
    model = Sequential()
    model.add(Conv2D(100, (3, 3), activation='relu', input_shape=(matrix_size,matrix_size, 1)))
    model.add(Conv2D(100, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    
    learning_rate = 0.05
    optimizer = SGD(lr=learning_rate, momentum=0.0, decay=0.001, nesterov=False)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.summary()
    
    if model_file != "" :
       model = keras.models.load_model(model_file)
       print("Realoded model from " + model_file)
    else:
       # early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, verbose=2, mode='min')
        history = model.fit(data2, labels, validation_data=[valid_data2, valid_label], 
                        epochs=1000, batch_size=100, verbose=2, shuffle='batch')#, callbacks=[early_stop])
            
        model.save("./model_output.hdf5", True)
 #       plot.plot(history.history['loss'])
 #       plot.plot(history.history['val_loss'])
 #       plot.title('loss, lr=' + str(learning_rate))
 #       plot.ylabel('loss')
 #       plot.xlabel('epoch')
 #       plot.legend(['training', 'validation'], loc='upper left')
 #       plot.show()
    
    num_true = 0
    for i in range(valid_data2.shape[0]):
        sample = np.reshape(valid_data2[i], (1,) + valid_data2[i].shape)
        pred  = model.predict(sample)
        if valid_label[i][0] == 0 :
            if pred[0][0] < 0.5 :
                num_true = num_true + 1
        else:
            if pred[0][0] >= 0.5 :
                num_true = num_true + 1
        if i < 10 :
            print("Prediction: " + str(pred) + " vs truth: " + str(labels[i]))
    
    print("Found " + str(num_true*100 / valid_label.shape[0]) + "% of correct samples")
    
