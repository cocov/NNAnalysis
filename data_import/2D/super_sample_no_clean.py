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
from ctapipe.image import hillas_parameters, tailcuts_clean, dilate

#draw neighbors
#from matplotlib.collections import PatchCollection, LineCollection

import numpy as np

#needed to create our own camera geometry for visualization
from astropy import units as u

import math
from matplotlib import pyplot as plt

import sys
from sys import stdout

import copy

import h5py 

#clean image based on timing. 
#@param surviving_pixels the boolean mask of surviving pixels. Initialy, only the pixel with the max. amplitude
#@param this_pixel the id of the current pixel
#@param values the array of integrated values of the pixels
#@param time_of_max array of time of maximum of the pulse for each pixel
#@param cam_geom the geometry of the camera
#@param base_time the time of the maximum of the reference pixel
#@param average the value of the average integrated value over the camera
def keep_similar_times_neighbors(surviving_pixels, this_pixel, values, time_of_max, cam_geom, base_time, average):

    max_delta_time = 6
    for n in cam_geom.neighbors[this_pixel]:
        if surviving_pixels[n] == False and math.fabs(time_of_max[n] - base_time) < max_delta_time:
            surviving_pixels[n] = True
            surviving_pixels = keep_similar_times_neighbors(surviving_pixels, n, values, time_of_max, cam_geom, base_time, average)
    return surviving_pixels



def compute_closest_neighbor(cam_geom, x, y):
    min_id = 0
    min_distance_sqr = 10000000
    
    for p in cam_geom.pix_id:
        d = (x - cam_geom.pix_x[p].value)*(x - cam_geom.pix_x[p].value) + (y - cam_geom.pix_y[p].value)*(y - cam_geom.pix_y[p].value)
        
        if d < min_distance_sqr:
            min_id = p
            min_distance_sqr = d
            
    return {'id':min_id, 'dist_sqr':min_distance_sqr}


def squared_distance(x1,y1,x2,y2):
    return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)


remapping_computed = False
closest_neighbors  = {}

#compute the square values of the camera
#@param cam_geom the source geometry
#@param square_value a numpy 1D array storing the values of the square image
#@param matrix_size the number of row (or columns) of the square image
def compute_square_values(cam_geom, cam_values, square_values, matrix_size):
    
    #calculate center of camera
    average_x = 0.0
    average_y = 0.0
    max_x = -10000
    max_y = -10000
    min_x = 10000
    min_y = 10000
    num_pixels = 1764
    
    for i in cam_geom.pix_x:
        average_x = average_x + i.value
        if i.value > max_x :
            max_x = i.value
        if i.value < min_x:
            min_x = i.value 
            
    for i in cam_geom.pix_y:
        average_y = average_y + i.value
        if i.value > max_y:
            max_y = i.value
        if i.value < min_y:
            min_y = i.value
            
    average_x = average_x / num_pixels
    average_y = average_y / num_pixels 
    
    if math.fabs(average_x) > 0.00001:
        print("ERROR: CAMERA NOT CENTERED AROUND 0 for X axis")
    if math.fabs(average_y) > 0.00001:
        print("ERROR: CAMERA NOT CENTERED AROUND 0 for Y axis")
    
    min = min_x
    if min_y < min:
        min = min_y
    max = max_x
    if max_y > max:
        max = max_y
        
    #need 1.1 scaling factor otherwise it leaves blank spaces between pixels
    #need 0.5 scaling factor to obtain radius rather than diameter
    pixel_radius = 1.2 * 0.5* math.sqrt((cam_geom.pix_x[cam_geom.pix_id[0]].value - cam_geom.pix_x[cam_geom.neighbors[cam_geom.pix_id[0]][0]].value)*(cam_geom.pix_x[cam_geom.pix_id[0]].value - cam_geom.pix_x[cam_geom.neighbors[cam_geom.pix_id[0]][0]].value) + (cam_geom.pix_y[cam_geom.pix_id[0]].value - cam_geom.pix_y[cam_geom.neighbors[cam_geom.pix_id[0]][0]].value)*(cam_geom.pix_y[cam_geom.pix_id[0]].value - cam_geom.pix_y[cam_geom.neighbors[cam_geom.pix_id[0]][0]].value))
    
    min = min - pixel_radius
    max = max + pixel_radius
    
    max = max * 1.2
    min = min * 1.2
    
    square_pixel_size = (max - min) / matrix_size
    
    min = min - square_pixel_size / 2.0
    max = max + square_pixel_size / 2.0
    
    global remapping_computed 
    global closest_neighbors  

    if remapping_computed == False:
        
        #init optimization with insane values to prevent pixels out of the camera from being ever updated
        for i in range(0, matrix_size):
            for j in range(0, matrix_size):
                closest_neighbors[i*matrix_size + j] = {'id':-1, 'dist_sqr':10000000}
        
        #we will in the matrix values, but starting from cherenkov camera pixel to reduce the complexity (hopefully)
        for p in cam_geom.pix_id:
            
            stdout.write("\rPrecomputing pix %d" % p)
            #compute the cell overlaid to the center of this pixel
            cell_i = math.floor((cam_geom.pix_x[p].value - min) / square_pixel_size)
            cell_j = math.floor((cam_geom.pix_y[p].value - min) / square_pixel_size)
    
            square_values[cell_i*matrix_size + cell_j] = cam_values[p]
            closest_neighbors[cell_i*matrix_size + cell_j] = {'id':p, 'dist_sqr':0}
            
            #now crawl the surrounding matrix cells and see if they belong to here or rather to a neighbor
            belongs_to_p = True
            distance     = 0
            #keep doing it until no more cell fall into this cherenkov pixel
            while belongs_to_p == True:
                distance = distance + 1
                belongs_to_p = False
                for x in range(-distance, distance+1):
                    for y in range(-distance, distance+1):
                        
                        if math.fabs(x) < distance and math.fabs(y) < distance:
                            continue #we've already done these guys earlier on
                        
                        this_cell_i = cell_i + x
                        this_cell_j = cell_j + y
                        if (this_cell_i < 0 or this_cell_i >= matrix_size or 
                            this_cell_j < 0 or this_cell_j >= matrix_size):
                            continue
                        
                        #compute distance to current pixels
                        still_in_cell = True
                        this_cell_x = min + this_cell_i*square_pixel_size + square_pixel_size/2.0
                        this_cell_y = min + this_cell_j*square_pixel_size + square_pixel_size/2.0
                        dist2 = squared_distance(this_cell_x, this_cell_y, cam_geom.pix_x[p].value, cam_geom.pix_y[p].value)
                        
                        #make sure that we are not outside of the cherenkov camera
                        if dist2 > pixel_radius*pixel_radius:
                            continue
                        
                        #compute distance to neighbor cherenkov pixels
                        for n in cam_geom.neighbors[p]:
                            dist2n = squared_distance(this_cell_x, this_cell_y, cam_geom.pix_x[n].value, cam_geom.pix_y[n].value)
                            if dist2n < dist2:
                                still_in_cell = False
                                
                        if still_in_cell:
                            belongs_to_p = True
                            square_values[this_cell_i*matrix_size + this_cell_j] = cam_values[p]
                            closest_neighbors[this_cell_i*matrix_size + this_cell_j] = {'id':p, 'dist_sqr':dist2}
    else:                
                    
        for x_id in range(0, matrix_size):
            if remapping_computed == False:
                print("Doing " + str(x_id) )
            for y_id in range(0, matrix_size):
                x = min + x_id*square_pixel_size 
                y = min + y_id*square_pixel_size
                if remapping_computed == True:
                    closest_neighbor = closest_neighbors[x_id*matrix_size + y_id]
                else:    
                    closest_neighbor = compute_closest_neighbor(cam_geom, x, y)
                    closest_neighbors[x_id*matrix_size + y_id] = closest_neighbor
                if (closest_neighbor['dist_sqr'] < 5 * pixel_radius * pixel_radius): #FIXME I have no idea why I needed to add a scaling factor here
                    square_values[x_id*matrix_size + y_id] = cam_values[closest_neighbor['id']]
    #            else:
    #                print("Radius: " + str(pixel_radius*pixel_radius) + " distance: " + str(closest_neighbor['dist_sqr']))
        
    remapping_computed = True
    return

if __name__ == '__main__':
    
    opts_parser = OptionParser()
    opts_parser.add_option("-i", "--input", dest="input", help="input hdf5 file", default="", type=str)
    opts_parser.add_option("-w", "--which", dest="which_one", help="either display square (0) or hexagonal(1)", default="0", type=int)
    opts_parser.add_option("-o", "--output", dest="output", help="Write data to outputfile rather than display it", default="", type=str)
    
    
    sys.setrecursionlimit(10000)

    (options, args) = opts_parser.parse_args()
    
    input_file = options.input
    which_one  = options.which_one
    output_filename = options.output
    
    if output_filename != "":
        print("Writing super-sampled data from " + input_file + " to " + output_filename)

        #get input data
        data   = HDF5Matrix(input_file, 'traces')
        tiom   = HDF5Matrix(input_file, 'tiom')
        labels = HDF5Matrix(input_file, 'labels')
        
        #compute square geometry
        cam_geom        = CameraGeometry.from_name("FlashCam")

        matrix_size = 299
        num_values = matrix_size * matrix_size
        square_values   = np.empty((matrix_size, matrix_size, 3))
        linear_square_values = np.empty(matrix_size*matrix_size)
        
        for i in range(0, matrix_size):
            for j in range(0, matrix_size):
                square_values[i][j][2] = 0.0
                
        #open outputs
        out_file = h5py.File(output_filename, 'w')
        out_traces = out_file.create_dataset('tracesandtiming', (100, matrix_size, matrix_size, 3), maxshape=(None, matrix_size, matrix_size, 3), dtype='i2')
        out_labels = out_file.create_dataset('labels', (100, 2), maxshape=(None, 2), dtype='i2')
        
        event_number = 0
        #loop through events, squash, write
        while True:
            try:
                this_data   = data[event_number]
                this_timing = tiom[event_number]
                this_label  = labels[event_number]
                out_traces.resize(event_number+1, axis=0)
                out_labels.resize(event_number+1, axis=0)
                
                linear_square_values = np.empty(matrix_size*matrix_size)
                linear_square_values.fill(0)

                compute_square_values(cam_geom, this_data, linear_square_values, matrix_size)
                for i in range(0, matrix_size):
                    for j in range(0, matrix_size):
                        square_values[i][j][0] = linear_square_values[i*matrix_size + j]

                linear_square_values.fill(0)
                
                compute_square_values(cam_geom, this_timing, linear_square_values, matrix_size)
                
                for i in range(0, matrix_size):
                    for j in range(0, matrix_size):
                        square_values[i][j][1] = linear_square_values[i*matrix_size + j]

                out_traces[event_number, ] = square_values
                out_labels[event_number, ] = this_label
                event_number = event_number + 1
                stdout.write("\rWriting event %d" % event_number)
                stdout.flush()
            except IndexError:
                out_file.close()
                print("\nDone")
                break
