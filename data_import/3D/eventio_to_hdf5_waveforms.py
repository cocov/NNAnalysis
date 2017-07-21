#!/Users/lyard/anaconda3/envs/ctapipe/bin/python

from ctapipe.io.hessio import hessio_event_source
from pyhessio import close_file as close_hessio
from ctapipe.utils import get_dataset

from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

import numpy as np
import h5py

from keras.utils.io_utils import HDF5Matrix

from optparse import OptionParser

from sys import stdout

import os

import math

gamma_end  = False
hadron_end = False

#read 2 eventio files from MC production, one hadron and one gamma, extract events data and merge them into a hdf5 file
if __name__ == '__main__':
    
    opts_parser = OptionParser()
    opts_parser.add_option("-i", "--input", dest="input", help="input eventio file", default="", type=str)
    opts_parser.add_option("-o", "--output", dest="output", help="Output file", default="./output.hdf5", type=str)
    opts_parser.add_option("-n", "--num_evts", dest="num_evts", help="Num events to write", default=100000000, type=int)
    opts_parser.add_option("-s", "--skip", dest="skip_evt", help="Skip the N first events", default="0", type=int)
    opts_parser.add_option("-e", "--energy", dest="energy", help="Filter by energy band: 1=[0,500GeV[, 2=[500GeV, 2TeV[, 3=[2TeV,...]", default=0, type=int)
    
    #retrieve args
    (options, args) = opts_parser.parse_args()
    input_filename = options.input
    out_filename    = options.output
    target_num_evts = options.num_evts
    skip_events     = options.skip_evt
    energy          = options.energy
    
    print("Will write " + str(target_num_evts) + " events from " + input_filename  + " to " + out_filename )
    
    if energy != "":
        if energy == 1:
            print("Filtering by energy band: [0:0.31]TeV")
        if energy == 2:
            print("Filtering by energy band: [0.31:1]TeV")
        if energy == 3:
            print("Filtering by energy band: [1:...]TeV") 
    
    #change the number of pixels to target different cameras. Warning: some cameras have the same number of pixels (e.g. Nectar and LST)
    target_num_pixels = 1764
    
    this_file_labels = [1,0] #default label is gamma
    
    input_end = True
    if input_filename != "":
        input_end = False
        if input_filename.find("gamma") == -1:
            this_file_labels = [0,1]
    
    #list of telescope IDs corresponding to the desired camera
    tels_list  = []
    
    num_events  = 0
    skipped     = 0
    
    num_samples = 25
    
    #it seems that pyhessio does not allow to have move than one file open at a given time. 
    #This is annoying, so we will first extract the data to two separate hdf5 files, and merge later on
    #TODO erase .gtmp and .htmp file from the disk
    if input_end == False:
        #open intermediate file and create output arrays
        out_file = h5py.File(out_filename, 'w')
        traces   = out_file.create_dataset('traces', (100,target_num_pixels, num_samples),  maxshape=(None, target_num_pixels, num_samples), dtype='i2')
        energies = out_file.create_dataset('energy', (100, 1), maxshape=(None, 1), dtype='f')
        labels   = out_file.create_dataset('labels', (100, 2), maxshape=(None, 2), dtype='i2')    
        #open input hessio and get first array event
        events_source = hessio_event_source(input_filename)
        
        list_initialized = False 
        #loop over array events to extract data
        while input_end == False:
            
            try:
                if num_events >= target_num_evts:
                    break   
                event = next(events_source)           
            except StopIteration:
                break
            
            #figure out which telescope IDs correspond to the desired camera based on the number of pixels
            if list_initialized == False:
                for i in event.inst["telescope_ids"]:
                    if event.inst["num_pixels"][i] == target_num_pixels:
                        tels_list.append(i)
                list_initialized = True;
                
            found_telescope = False
            
            if energy != 0:
                if energy == 1 and event.mc.energy.value >= 0.31:
                    continue
                if energy == 2 and (event.mc.energy.value < 0.31 or event.mc.energy.value >= 1):
                    continue
                if energy == 3 and event.mc.energy.value < 1:
                    continue
                
            for key, value in event.r0.tel.items():
                if key in tels_list:
                    if found_telescope:
                        continue
                    found_telescope = True
                    if skipped < skip_events :
                        skipped = skipped + 1
                        stdout.write("\rSkipping evt: %d" % skipped)
                        continue
                    #add more room to the output file and write new data
                    traces.resize(  num_events+1, axis=0)
                    energies.resize(num_events+1, axis=0)
                    labels.resize(  num_events+1, axis=0)
                    
                    traces[num_events, ]   = value.adc_samples[0] 
                    energies[num_events, ] = energy
                    labels[num_events , ]  = this_file_labels
                    num_events = num_events + 1
                    #print(str((num_events-1)* 2 + 1) + " " + str(event.mc.energy))
                    stdout.write("\rWriting evt. %d" % num_events)
                    if num_events >= target_num_evts:
                        break   

        close_hessio()
        out_file.close()
        print("\nDone writing Events")
            
    print("Done.")
