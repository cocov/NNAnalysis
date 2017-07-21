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

import math

import sys
from sys import stdout

import h5py 

#needed to allow the recursion to run. Otherwise, stack overflow !
sys.setrecursionlimit(10000)

#recursively compute the pixels location in the square matrix
# @param real_geom the input geometry from ctapipe
# @param current_dict the new mapping computed up to now
# @param current_cell the id of the pixel we're currently in
# WARNING: one assumption is made about the orientation: the flats of the pixels is "up and down", not "right and left"
# this means that the neighbors are noth, south, north-east, south-east, north-west and south-west
# the other orientation would give neighbors right, left, north-east, north-west, south-east and south-west
def recurse_build_square_geometry(real_geom, current_dict, current_cell):
    
    #we are not too sure about the ordering of the neighbors, so let's double check using pixels coordinates
    x = real_geom.pix_x[current_cell].value
    y = real_geom.pix_y[current_cell].value
    
    #crawl through this pixel's neighbors
    for n in real_geom.neighbors[current_cell] :

        #if we've already added it to the new mapping, skip it        
        if n in current_dict:
            continue
        
        x2 = real_geom.pix_x[n].value
        y2 = real_geom.pix_y[n].value
       
        #if same x for the two pixels, either north or south pixel
        if math.fabs(x - x2) < 0.001 :
            if y < y2 : #north
                current_dict[n] = [ current_dict[current_cell][0] + 0, current_dict[current_cell][1] + 1 ]
            else:       #south
                current_dict[n] = [ current_dict[current_cell][0] + 0, current_dict[current_cell][1] - 1 ]
        
        #larger x for neighbor: east neighbors
        if x < x2 :
            if y < y2 : #north-east
                current_dict[n] = [ current_dict[current_cell][0] + 1, current_dict[current_cell][1] + 1 ]
            else:       #south-east
                current_dict[n] = [ current_dict[current_cell][0] + 1, current_dict[current_cell][1] + 0 ]
        
        #smaller x for neighbor: west neighbor
        if x > x2 :
            if y < y2 : #north-west
                current_dict[n] = [ current_dict[current_cell][0] - 1, current_dict[current_cell][1] + 0 ]
            else: #south-west
                current_dict[n] = [ current_dict[current_cell][0] - 1, current_dict[current_cell][1] - 1 ]
                
        #do it again, taking the current neighbor as center pixel
        current_dict = recurse_build_square_geometry(real_geom, current_dict, n)
 
    return current_dict

#Takes a regular geometry and returns its square equivalent
#pixels indices that are not used in the square camera image are labelled as ??
def build_square_geometry(real_geom):
    
    #crawl the camera geometry
    #1. build list of pixels neighbors. Done. Already from ctapipe
    print("Computing square geometry for "+real_geom.cam_id)

    cells_dict = {real_geom.pix_id[0] : [0,0]}
    
    square_geometry = recurse_build_square_geometry(real_geom, cells_dict, real_geom.pix_id[0])
    
    #figure out what is the actual size of the newly constructed matrix
    min_x = 100000
    min_y = 100000
    max_x = -100000
    max_y = -100000
    for key, value in square_geometry.items() :
        if value[0] < min_x :
            min_x = value[0]
        if value[0] > max_x :
            max_x = value[0]
        if value[1] < min_y :
            min_y = value[1]
        if value[1] > max_y :
            max_y = value[1]
        
    #print("Min max: x=[" + str(min_x) + ":" + str(max_x) + "] y=[" + str(min_y) + ":" + str(max_y) + "]")
    
    for key, value in square_geometry.items() :
        square_geometry[key] = [value[0] - min_x, value[1] - min_y]
    
    return square_geometry

if __name__ == '__main__':
    
    opts_parser = OptionParser()
    opts_parser.add_option("-i", "--input", dest="input", help="input hdf5 file", default="", type=str)
    opts_parser.add_option("-w", "--which", dest="which_one", help="either display square values (0), square times (1) or hexagonal values (2) or hexagonal times (3)", default="0", type=int)
    opts_parser.add_option("-o", "--output", dest="output", help="Write data to outputfile rather than display it", default="", type=str)
    opts_parser.add_option("-n", "--number", dest="max_num", help="number of events to process", default="10000000000", type=int)
    
    (options, args) = opts_parser.parse_args()
    
    input_file = options.input
    which_one  = options.which_one
    output_filename = options.output
    max_num_evts = options.max_num
    
    if output_filename != "":
        print("Writing hex-to-square data from " + input_file + " to " + output_filename)

        #get input data
        data   = HDF5Matrix(input_file, 'traces')
        labels = HDF5Matrix(input_file, 'labels')
        times  = HDF5Matrix(input_file, 'tiom')
        
        #compute square geometry
        cam_geom        = CameraGeometry.from_name("FlashCam")
        square_geometry = build_square_geometry(cam_geom)
        matrix_size = 56
        num_values = matrix_size * matrix_size
        square_values   = np.empty((matrix_size, matrix_size, 2))
        square_values.fill(0)
        
        #open outputs
        out_file = h5py.File(output_filename, 'w')
        out_traces = out_file.create_dataset('tracesandtiming', (100, matrix_size, matrix_size, 2), maxshape=(None, matrix_size, matrix_size, 2), dtype='i2')
        out_labels = out_file.create_dataset('labels', (100, 2), maxshape=(None, 2), dtype='i2')
        
        event_number = 0

        #loop through events, squash, write
        while True:
            try:
                this_data   = data[event_number]
                this_timing = times[event_number]
                this_label  = labels[event_number]
                
                out_traces.resize(event_number+1, axis=0)
                out_labels.resize(event_number+1, axis=0)
                
                #assign the input values to the output matrix
                for key, value in square_geometry.items() :
                    square_values[value[0]][value[1]][0] = this_data[key]
                    square_values[value[0]][value[1]][1] = this_timing[key]
                    
                out_traces[event_number, ] = square_values
                out_labels[event_number, ] = this_label
                event_number = event_number + 1
                stdout.write("\rWriting event %d" % event_number)
                if event_number >= max_num_evts:
                    break
                
                stdout.flush()
            except IndexError:
                out_file.close()
                print("\nDone")
                break
        exit(0)
    
    print("Displaying data from " + input_file)
    if which_one == 0 or which_one == 1:
        reread_data = HDF5Matrix(input_file, 'tracesandtiming')
    else:
        if which_one == 2:
            reread_data = HDF5Matrix(input_file, 'traces')
        else:
            reread_data = HDF5Matrix(input_file, 'tiom')
            
    reread_labels = HDF5Matrix(input_file, 'labels')
    stop = ""
    event_number = 0
    cam_geom = CameraGeometry.from_name("FlashCam")
    
    print(cam_geom.pix_x.__class__)
    
    square_geometry = build_square_geometry(cam_geom)
    
    #create a square geometry
    matrix_size       = 56
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

    if which_one == 0 or which_one == 1:
        disp = CameraDisplay(square_camera)    
        disp.add_colorbar()
    else:
        disp = CameraDisplay(cam_geom)
        disp.add_colorbar()
    
    square_values = np.empty(matrix_size*matrix_size)
    square_values.fill(0)
    
    for key, value in square_geometry.items() :
        square_values[value[0]*matrix_size + value[1]] = key
    
    while stop != "q":
        try:
            if which_one == 0 or which_one == 1:
                for i in range(0, matrix_size):
                    for j in range(0, matrix_size):
                        square_values[i*matrix_size + j] = reread_data[event_number][i][j][which_one]
            
            if which_one == 0 or which_one == 1:
                disp.image = square_values #reread_data[event_number]
                disp.show()
            else:
                disp.image = reread_data[event_number]
                disp.show()
            print(reread_labels[event_number])
            if reread_labels[event_number][0] == 1:
                print ("Gamma")
            else:
                print ("Hadron")

            event_number = event_number + 1
           
            stop = input('\'q\' to quit:')
        except IndexError:
            print ("Reached end-of-file")
            stop = "q"
        