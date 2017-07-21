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

import math

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
    opts_parser.add_option("-s", "--skip", dest="skip_evt", help="Skip the N first events", default="0", type=int)
    opts_parser.add_option("-e", "--energy", dest="energy", help="Filter by energy band: 1=[0,500GeV[, 2=[500GeV, 2TeV[, 3=[2TeV,...]", default=0, type=int)
    
    #retrieve args
    (options, args) = opts_parser.parse_args()
    hadron_filename = options.hadron
    gamma_filename  = options.gamma
    out_filename    = options.output
    target_num_evts = options.num_evts
    display_data    = options.display
    skip_events     = options.skip_evt
    energy          = options.energy
    
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
    if gamma_filename == "":
        gamma_end = True
    if hadron_filename == "":
        hadron_end = True
    if energy != "":
       print("Filtering by energy band: " + str(energy)) 
    
    #change the number of pixels to target different cameras. Warning: some cameras have the same number of pixels (e.g. Nectar and LST)
    target_num_pixels = 1764
    
    #list of telescope IDs corresponding to the desired camera
    gamma_tels_list  = []
    hadron_tels_list = []
    
    num_events  = 0
    num_gammas  = 0
    num_hadrons = 0
    skipped     = 0
    
    samples_around_time_of_max = 4
    
    #it seems that pyhessio does not allow to have move than one file open at a given time. 
    #This is annoying, so we will first extract the data to two separate hdf5 files, and merge later on
    #TODO erase .gtmp and .htmp file from the disk
    if gamma_end == False:
        #open intermediate file and create output arrays
        out_file = h5py.File(out_filename+".gtmp", 'w')
        traces = out_file.create_dataset('traces', (100,target_num_pixels),  maxshape=(None, target_num_pixels), dtype='i2')
        tiom   = out_file.create_dataset('tiom', (100, target_num_pixels), maxshape=(None, target_num_pixels), dtype='i2')
        labels = out_file.create_dataset('labels', (100, 2), maxshape=(None, 2), dtype='i2')    
        #open input hessio and get first array event
        gamma_source = hessio_event_source(gamma_filename)
        
        list_initialized = False 
        #loop over array events to extract data
        while gamma_end == False:
            
            try:
                if num_events >= target_num_evts/2:
                    break   
                event = next(gamma_source)           
            except StopIteration:
                break
            
            #figure out which telescope IDs correspond to the desired camera based on the number of pixels
            if list_initialized == False:
                for i in event.inst["telescope_ids"]:
                    if event.inst["num_pixels"][i] == target_num_pixels:
                        gamma_tels_list.append(i)
                list_initialized = True;
                
            found_telescope = False
            
            if energy != 0:
                if energy == 1 and event.mc.energy.value >= 0.5:
                    continue
                if energy == 2 and (event.mc.energy.value < 0.5 or event.mc.energy.value >= 2):
                    continue
                if energy == 3 and event.mc.energy.value < 2:
                    continue
                
            for key, value in event.r0.tel.items():
                if key in gamma_tels_list:
                    if found_telescope:
                        continue
                    found_telescope = True
                    if skipped < skip_events/2 :
                        skipped = skipped + 1
                        stdout.write("\rSkipping evt: %d" % skipped)
                        continue
                    #add more room to the output file and write new data
                    traces.resize(num_events+1, axis=0)
                    tiom.resize(num_events+1, axis=0)
                    labels.resize(num_events+1, axis=0)
                    
                    #calculate time of max for all pixels. 
                    time_of_max = np.empty(target_num_pixels)
                    for i in range(0,target_num_pixels):
                        max            = 0
                        time_of_max[i] = 0
                        for j in range(0,value.num_samples):
                            if value.adc_samples[0][i][j] > max:
                                max = value.adc_samples[0][i][j]
                                time_of_max[i] = j
                            
                    traces[num_events, ] = np.squeeze(value.adc_sums) 
                    tiom[num_events, ]   = time_of_max 
                    labels[num_events ,] = [1,0] #gamma flag
                    num_events = num_events + 1
                    num_gammas = num_gammas + 1
                    #print(str((num_events-1)* 2 + 1) + " " + str(event.mc.energy))
                    stdout.write("\rWriting Gamma evt. %d" % num_events)
                    if num_events >= target_num_evts/2:
                        break   

        close_hessio()
        out_file.close()
        print("\nDone writing Gammas")
    num_events = 0
    skipped = 0
    #do the same thing as before (just above) with the hadron file
    if hadron_end == False:
        out_file = h5py.File(out_filename+".htmp", 'w')
        traces = out_file.create_dataset('traces', (100,target_num_pixels),  maxshape=(None, target_num_pixels), dtype='i2')
        tiom   = out_file.create_dataset('tiom', (100, target_num_pixels), maxshape=(None, target_num_pixels), dtype='i2')
        labels = out_file.create_dataset('labels', (100, 2), maxshape=(None, 2), dtype='i2')    
        hadron_source = hessio_event_source(hadron_filename)

        list_initialized = False  
        while hadron_end == False:
            #get a new event
            try:
                if num_events >= target_num_evts/2:
                    break
                event = next(hadron_source)
            except StopIteration:
                break
            
            #if not already done, initialize the list of events of interest
            if list_initialized == False:
                for i in event.inst["telescope_ids"]:
                    if event.inst["num_pixels"][i] == target_num_pixels:
                        hadron_tels_list.append(i)      
                list_initialized = True
                
            #make sure that we read one and only one event for this type of telescope in this array event
            found_telescope = False
            
            #filter by energy if needed
            if energy != 0:
                if energy == 1 and event.mc.energy.value >= 0.5:
                    continue
                if energy == 2 and (event.mc.energy.value < 0.5 or event.mc.energy.value >= 2):
                    continue
                if energy == 3 and event.mc.energy.value < 2:
                    continue
    
            for key, value in event.r0.tel.items():
                if key in hadron_tels_list:
                    if found_telescope:
                        continue
                    found_telescope = True
                    if skipped < skip_events/2 :
                        skipped = skipped + 1
                        stdout.write("\rSkipping evt: %d" % skipped)
                        continue
                    traces.resize(num_events+1, axis=0)
                    tiom.resize(num_events+1, axis=0)
                    labels.resize(num_events+1, axis=0)

                    #calculate time of max for all pixels. 
                    #also extracts time of global max on the way
                    time_of_max = np.empty(target_num_pixels)
                    for i in range(0,target_num_pixels):
                        max            = 0
                        time_of_max[i] = 0
                        for j in range(0,value.num_samples):
                            if value.adc_samples[0][i][j] > max:
                                max = value.adc_samples[0][i][j]
                                time_of_max[i] = j

                    traces[num_events, ] = np.squeeze(value.adc_sums) 
                    tiom[num_events, ]   = time_of_max 
                    labels[num_events ,] = [0,1] #hadron flag
                    num_events = num_events + 1
                    num_hadrons = num_hadrons + 1
                    #print(str(num_events*2) + " " + str(event.mc.energy))
                    stdout.write("\rWriting Hadron evt. %d" % num_events)
                    if num_events >= target_num_evts/2:
                        break

        close_hessio()
        out_file.close()
        print("\nDone writing Hadrons")
    #now we've written the events separately in two hdf5 files. reopen these files to merge them
    num_events  = 0
    num_hadrons = 0
    num_gammas  = 0
    num_src_events = 0
    if (gamma_end == False) or (hadron_end == False):
        out_file = h5py.File(out_filename, 'w')
        traces = out_file.create_dataset('traces', (100,target_num_pixels),  maxshape=(None, target_num_pixels), dtype='i2')
        tiom   = out_file.create_dataset('tiom', (100, target_num_pixels), maxshape=(None, target_num_pixels), dtype='i2')
        labels = out_file.create_dataset('labels', (100, 2), maxshape=(None, 2), dtype='i2')
        hadrons = HDF5Matrix(out_filename+".htmp", 'traces')
        gammas  = HDF5Matrix(out_filename+".gtmp", 'traces')
        hadrons_tiom = HDF5Matrix(out_filename+".htmp", 'tiom')
        gammas_tiom  = HDF5Matrix(out_filename+".gtmp", 'tiom')
        while True:
            try:
                this_hadron = hadrons[num_src_events]
                this_gamma  = gammas[num_src_events]    
                this_hadron_tiom = hadrons_tiom[num_src_events]
                this_gamma_tiom  = gammas_tiom[num_src_events]
                traces.resize(num_events+2, axis=0)
                tiom.resize(num_events+2,   axis=0)
                labels.resize(num_events+2, axis=0)
                traces[num_events, ]   = np.squeeze(this_hadron)
                traces[num_events+1, ] = np.squeeze(this_gamma)
                tiom[num_events, ]   = np.squeeze(this_hadron_tiom)
                tiom[num_events+1, ] = np.squeeze(this_gamma_tiom)
                labels[num_events, ]   = [0,1]
                labels[num_events+1, ] = [1,0]
                num_events  = num_events + 2
                num_src_events = num_src_events + 1
                num_gammas  = num_gammas + 1
                num_hadrons = num_hadrons + 1
                stdout.write("\rMerging evt. %d" % num_events)
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
                print(reread_labels[event_number])
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
