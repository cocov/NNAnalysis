'''
Created on 30 Jun 2017

@author: lyard
'''
#run from command line
from optparse import OptionParser

#read/write keras objects
from keras.utils.io_utils import HDF5Matrix

from sys import stdout

import h5py

if __name__ == '__main__':
    opts_parser = OptionParser()
    opts_parser.add_option("-g", "--gamma", dest="gammas", help="text file containing a list of input HDF5 supersampled camera images", default="", type=str)
    opts_parser.add_option("-p", "--proton", dest="protons", help="text file containing a list of input HDF5 supersampled camera images", default="", type=str)
    opts_parser.add_option("-o", "--output", dest="output", help="name of output file", default="", type=str)
    opts_parser.add_option("-n", "--num_evts", dest="number", help="number of events to write to output", default=10000000, type=int)
    
    (options, args) = opts_parser.parse_args()
    
    input_gamma_list_filename  = options.gammas
    input_proton_list_filename = options.protons
    output_filename   = options.output
    max_num_events    = options.number
    
    if input_gamma_list_filename == "":
        print("Please provide an input list of gamma files")
        exit(-1)
    if input_proton_list_filename == "":
        print("Please provide an input list of proton files")
        exit(-1)
    if output_filename == "":
        print("Please provide an output filename")
        exit(-1)
        
    #create an array from the files list
    input_gamma_list_file = open(input_gamma_list_filename, "r")
    input_proton_list_file = open(input_proton_list_filename, "r")
    
    input_gamma_list = input_gamma_list_file.readlines()
    input_proton_list = input_proton_list_file.readlines()
    
    #remove trailing \n from lists entries
    for item in range(0,len(input_gamma_list)-1):
        input_gamma_list[item] = input_gamma_list[item][:-1]
    for item in range(0, len(input_proton_list)-1):
        input_proton_list[item] = input_proton_list[item][:-1]

    num_gamma_files  = len(input_gamma_list) - 1
    num_proton_files = len(input_proton_list) - 1

    #make sure that the 'end-' tag is here
    for item in range(0, len(input_gamma_list)-1):
        if input_gamma_list[item] == 'end':
            num_gamma_files = item
        else:
            print(input_gamma_list[item])
    for item in range(0, len(input_proton_list)-1):
        if input_proton_list[item] == 'end':
            num_proton_files = item
            
    #make sure that all input files indeed exist
    for item in range(0, len(input_gamma_list)-1):
        test_file = open(input_gamma_list[item], "r")
        test_file.close()
    for item in range(0, len(input_proton_list)-1):
        test_file = open(input_proton_list[item], "r")
        test_file.close()

    print("Will use " + str(num_gamma_files) + " gamma input files and " + str(num_proton_files) + " protons")
    
    matrix_size = 56
    
    #open output file
    output_file = h5py.File(output_filename, 'w')
    out_traces = output_file.create_dataset('traces', (100, matrix_size, matrix_size, 25, 1), maxshape=(None, matrix_size, matrix_size, 25, 1), dtype='i2')
    out_labels = output_file.create_dataset('labels', (100, 2), maxshape=(None, 2), dtype='i2')
    
    #open first available input files
    gamma_file_index  = 0
    proton_file_index = 0
    gamma_evt_index  = 0
    proton_evt_index = 0
   
    print("Starting with " + input_gamma_list[gamma_file_index] + " and " + input_proton_list[proton_file_index])
    #f = h5py.File(input_gamma_list[gamma_file_index], 'r')
    #print(input_gamma_list[gamma_file_index])
    #print(f.__contains__('tracesandtiming'))
    #print(f[f.keys()[0]])
    gamma_data = HDF5Matrix(input_gamma_list[gamma_file_index], 'traces')
    proton_data = HDF5Matrix(input_proton_list[proton_file_index], 'traces')
    
    num_events_written = 0
    
    while num_events_written < max_num_events:
        #get a new gamma
        try:
            this_gamma = gamma_data[gamma_evt_index]
        except IndexError:
            #try to get another file
            gamma_file_index = gamma_file_index + 1
            if gamma_file_index < num_gamma_files:
                print(" --- Moving to a new gamma file: " + input_gamma_list[gamma_file_index])
                gamma_data = HDF5Matrix(input_gamma_list[gamma_file_index], 'traces')
                gamma_evt_index = 0
            else:
                print(" --- Exausted gamma data: finishing")
                break
        
        #get a new proton
        try:
            this_proton = proton_data[proton_evt_index]
        except IndexError:
            proton_file_index = proton_file_index + 1
            if proton_file_index < num_proton_files:
                print(" --- Moving to a new proton file: " + input_proton_list[proton_file_index])
                proton_data = HDF5Matrix(input_proton_list[proton_file_index], 'traces')
                proton_evt_index = 0
            else:
                print(" --- Exausted proton data: finishing")
                break
        
        #we got a new set of two events to write
        out_traces.resize(num_events_written + 2, axis=0)
        out_labels.resize(num_events_written + 2, axis=0)
        
        out_traces[num_events_written + 0, ] = this_gamma
        out_traces[num_events_written + 1, ] = this_proton
        out_labels[num_events_written + 0, ] = [1, 0]
        out_labels[num_events_written + 1, ] = [0, 1]
        
        num_events_written  = num_events_written + 2  
        gamma_evt_index = gamma_evt_index + 1
        proton_evt_index = proton_evt_index + 1        

        stdout.write("\rWrote %d evts" % num_events_written)
        
        if num_events_written >= max_num_events:
            break
        
    print("All done")
    
    output_file.close()
            
                
                
                
                
