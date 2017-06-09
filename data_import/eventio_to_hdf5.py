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

gamma_end  = False
hadron_end = False

#read 2 eventio files from MC production, one hadron and one gamma, extract events data and merge them into a hdf5 file
if __name__ == '__main__':
    
    opts_parser = OptionParser()
    opts_parser.add_option("-p", "--hadron", dest="hadron", help="Input hadronic data", default="", type=str)
    opts_parser.add_option("-g", "--gamma", dest="gamma", help="Input gamma data", default="", type=str)
    opts_parser.add_option("-o", "--output", dest="output", help="Output file", default="./output.hdf5", type=str)
    opts_parser.add_option("-n", "--num_evts", dest="num_evts", help="Num events to write", default=100000000, type=int)
    opts_parser.add_option("-d", "--display", dest="display", action="store_true", help="Display written data", default=False)
    
    #retrieve args
    (options, args) = opts_parser.parse_args()
    hadron_filename = options.hadron
    gamma_filename  = options.gamma
    out_filename    = options.output
    target_num_evts = options.num_evts
    display_data    = options.display
    
    print("Will write " + str(target_num_evts) + " events from " + hadron_filename + " and " + gamma_filename + " to " + out_filename )
    
    #make sure that we will be doing something
    if hadron_filename == "" and gamma_filename != "":
        print("Missing hadron data input file")
        exit(-1)
    if gamma_filename == "" and hadron_filename != "":
        print("Missing gamma data input file")
        exit(-1)
    if (gamma_filename == "" and hadron_filename == "" and display_data == False):
        print("Please do something: generate hdf5 or display merged data")
        exit(-1)
    if (gamma_filename == ""):
        gamma_end = True
    if (hadron_filename == ""):
        hadron_end = True
    
    #change the number of pixels to target different cameras. Warning: some cameras have the same number of pixels (e.g. Nectar and LST)
    target_num_pixels = 1764
    
    #list of telescope IDs corresponding to the desired camera
    gamma_tels_list  = []
    hadron_tels_list = []
    
    num_events  = 0
    num_gammas  = 0
    num_hadrons = 0
    
    #it seems that pyhessio does not allow to have move than one file open at a given time. 
    #This is annoying, so we will first extract the data to two separate hdf5 files, and merge later on
    #TODO erase .gtmp and .htmp file from the disk
    if gamma_end == False:
        #open intermediate file and create output arrays
        out_file = h5py.File(out_filename+".gtmp", 'w')
        traces = out_file.create_dataset('traces', (100,target_num_pixels),  maxshape=(None, target_num_pixels), dtype='i2')
        labels = out_file.create_dataset('labels', (100, 2), maxshape=(None, 2), dtype='i2')    
        #open input hessio and get first array event
        gamma_source = hessio_event_source(gamma_filename)
        event = next(gamma_source)
        #figure out which telescope IDs correspond to the desired camera based on the number of pixels
        for i in event.inst["telescope_ids"]:
            if event.inst["num_pixels"][i] == target_num_pixels:
                gamma_tels_list.append(i)
        #loop over array events to extract data
        while gamma_end == False:
            for key, value in event.r0.tel.items():
                if key in gamma_tels_list:
                    #add more room to the output file and write new data
                    traces.resize(num_events+1, axis=0)
                    labels.resize(num_events+1, axis=0)
                    traces[num_events, ] = np.squeeze(value.adc_sums)
                    labels[num_events ,] = [1,0] #gamma flag
                    num_events = num_events + 1
                    num_gammas = num_gammas + 1
                    stdout.write("\rWriting Gamma evt. %d" % num_events)
                    if num_events >= target_num_evts/2:
                        break   
            try:
                if num_events >= target_num_evts/2:
                    break   
                event = next(gamma_source)    
            except StopIteration:
                break
        close_hessio()
        out_file.close()
        print("\nDone writing Gammas")
    num_events = 0
          
    #do the same thing as before (just above) with the hadron file
    if hadron_end == False:
        out_file = h5py.File(out_filename+".htmp", 'w')
        traces = out_file.create_dataset('traces', (100,target_num_pixels),  maxshape=(None, target_num_pixels), dtype='i2')
        labels = out_file.create_dataset('labels', (100, 2), maxshape=(None, 2), dtype='i2')    
        hadron_source = hessio_event_source(hadron_filename)
        event = next(hadron_source)
        for i in event.inst["telescope_ids"]:
            if event.inst["num_pixels"][i] == target_num_pixels:
                hadron_tels_list.append(i)        
        while hadron_end == False:
            for key, value in event.r0.tel.items():
                if key in hadron_tels_list:
                    traces.resize(num_events+1, axis=0)
                    labels.resize(num_events+1, axis=0)
                    traces[num_events, ] = np.squeeze(value.adc_sums)
                    labels[num_events ,] = [0,1] #hadron flag
                    num_events = num_events + 1
                    num_hadrons = num_hadrons + 1
                    stdout.write("\rWriting Hadron evt. %d" % num_events)
                    if num_events >= target_num_evts/2:
                        break
            try:
                if num_events >= target_num_evts/2:
                    break
                evevnt = next(hadron_source)
            except StopIteration:
                break
        close_hessio()
        out_file.close()
        print("\nDone writing Hadrons")
    #now we've written the events separately in two hdf5 files. reopen these files to merge them
    num_events  = 0
    num_hadrons = 0
    num_gammas  = 0
    
    if (gamma_end == False) or (hadron_end == False):
        out_file = h5py.File(out_filename, 'w')
        traces = out_file.create_dataset('traces', (100,target_num_pixels),  maxshape=(None, target_num_pixels), dtype='i2')
        labels = out_file.create_dataset('labels', (100, 2), maxshape=(None, 2), dtype='i2')
        hadrons = HDF5Matrix(out_filename+".htmp", 'traces')
        gammas  = HDF5Matrix(out_filename+".gtmp", 'traces')
        while True:
            try:
                this_hadron = hadrons[num_events]
                this_gamma  = gammas[num_events]    
                traces.resize(num_events+2, axis=0)
                labels.resize(num_events+2, axis=0)
                traces[num_events, ] = np.squeeze(this_hadron)
                traces[num_events+1, ] = np.squeeze(this_gamma)
                labels[num_events, ] = [0,1]
                labels[num_events+1, ] = [1,0]
                num_events  = num_events + 1
                num_gammas  = num_gammas + 1
                num_hadrons = num_hadrons + 1
                stdout.write("\rMerging evt. %d" % (num_events*2))
            except IndexError:
                out_file.close()
                break
                           
    if (gamma_end == False) or (hadron_end == False): 
        print("\nWrote " + str(num_gammas) + " gammas and " + str(num_hadrons) + " hadrons")
    
    if display_data == True:
        print("Displaying data from " + out_filename)
        reread_data   = HDF5Matrix(out_filename, 'traces')
        reread_labels = HDF5Matrix(out_filename, 'labels')
        stop = ""
        event_number = 0
        cam_geom = CameraGeometry.from_name("FlashCam")
        disp = CameraDisplay(cam_geom)
        disp.add_colorbar()
        while stop != "q":
            try:
                disp.image = reread_data[event_number]
                if reread_labels[event_number][0] == 1:
                    print ("Gamma")
                else:
                    print ("Hadron")
                event_number = event_number + 1
                disp.show()
                stop = input('\'q\' to quit:')
            except IndexError:
                print ("Reached end-of-file")
                stop = "q"
            
    print("Done.")
