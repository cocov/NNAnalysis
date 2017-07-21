'''
Created on 16 Jun 2017

@author: lyard
'''

'''
Created on 12 Jun 2017

@author: lyard
'''

#run from command line
from optparse import OptionParser

#read/write keras objects
from keras.utils.io_utils import HDF5Matrix

#look at the result
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

#draw neighbors
#from matplotlib.collections import PatchCollection, LineCollection

import numpy as np

#needed to create our own camera geometry for visualization
from astropy import units as u

if __name__ == '__main__':
    
    opts_parser = OptionParser()
    opts_parser.add_option("-i", "--input", dest="input", help="input hdf5 file", default="", type=str)
    opts_parser.add_option("-w", "--which", dest="which_one", help="either display amplitude (0) or timing(1)", default="0", type=int)
    
    (options, args) = opts_parser.parse_args()
    
    input_file = options.input
    which_one  = options.which_one
    
    print("Displaying data from " + input_file)
    data   = HDF5Matrix(input_file, 'tracesandtiming')        
    labels = HDF5Matrix(input_file, 'labels')
    stop = ""
    event_number = 0

    #create a square geometry
    matrix_size       = 299 #min size of inception NN
    square_num_pixels = matrix_size * matrix_size
    square_pix_ids = np.arange(square_num_pixels)
    square_x_pos   = np.empty(square_num_pixels).tolist()
    square_y_pos   = np.empty(square_num_pixels).tolist()
    pixel_size     = 0.01 #meters
    pixel_area     = np.empty(square_num_pixels).tolist()#pixel_size * pixel_size
    
    for i in range(0, matrix_size) :
        for j in range(0, matrix_size) :
            square_x_pos[i*matrix_size + j] = i*pixel_size
            square_y_pos[i*matrix_size + j] = j*pixel_size
            pixel_area[i*matrix_size + j] = pixel_size * pixel_size
            
    square_camera = CameraGeometry(cam_id   = "SquashCam", 
                                   pix_id   = square_pix_ids, 
                                   pix_x    = square_x_pos * u.meter, 
                                   pix_y    = square_y_pos * u.meter, 
                                   pix_area = pixel_area, 
                                   pix_type = 'rectangular')


    disp = CameraDisplay(square_camera)    
    disp.add_colorbar()
     
    #create square geometry for display purposes
    square_values = np.empty(matrix_size*matrix_size)
    square_values.fill(0)
    
    print(disp)
    
    while stop != "q":
        try:
            this_data = data[event_number]
            if which_one == 0:
                for i in range(0, matrix_size):
                    for j in range(0, matrix_size):
                        square_values[i*matrix_size + j] = this_data[i][j][0]
            else:
                for i in range(0, matrix_size):
                    for j in range(0, matrix_size):
                        square_values[i*matrix_size + j] = this_data[i][j][1]
            
            disp.image = square_values
            disp.show()
            
            if labels[event_number][0] == 1:
                print ("Gamma")
            else:
                print ("Hadron")

            event_number = event_number + 1
           
            stop = input('\'q\' to quit:')
        except IndexError:
            print ("Reached end-of-file")
            stop = "q"
        