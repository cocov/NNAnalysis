'''
Created on 22 Jun 2017

@author: lyard
'''

import matplotlib.pyplot as plt
import numpy as np
import sys 

#read the input energy per event
energy_file = open("energy_of_1k_events.txt", "r")
lines = energy_file.readlines()

events_energy_dict = {}

for line in lines:
    splitted = line.split(' ')
    events_energy_dict[int(splitted[0])] = float(splitted[1])
    
#read the results of the training
result_file = open("true_false_results.txt", "r")
lines = result_file.readlines()

results = np.empty(1001) #events are numbered starting from 1
this_item = 0

for line in lines:
    splitted = line.split(' ')
    this_item = this_item + 1
    if splitted[1] == "true\n":
        results[this_item] = True
        continue
    if splitted[1] == "FALSE-------\n":
        results[this_item] = False
        continue
    print("Unexpected entry: |" + splitted[1] + "|")

low_energy = [0,0] #[true, false]
mid_energy = [0,0]
hi_energy  = [0,0]

for i in range(1, 1001):
    if events_energy_dict[i] < 0.5:
        if results[i]:
            low_energy[0] = low_energy[0] + 1
        else:
            low_energy[1] = low_energy[1] + 1
    if events_energy_dict[i] >= 0.5 and events_energy_dict[i] < 2:
        if results[i]:
            mid_energy[0] = mid_energy[0] + 1
        else:
            mid_energy[1] = mid_energy[1] + 1  
    if events_energy_dict[i] >= 2:
        if results[i]:
            hi_energy[0] = hi_energy[0] + 1
        else:
            hi_energy[1] = hi_energy[1] + 1
            
print("Results:")
print("low - " + str(low_energy[0] / (low_energy[0] + low_energy[1])))
print("mid - " + str(mid_energy[0] / (mid_energy[0] + mid_energy[1])))
print("high- " + str(hi_energy[0] / (hi_energy[0] + hi_energy[1])))
    
exit(0)
#make a histogram !
energies = np.empty(1000)
min_val   = 10000 
max_val   = 0 
num_moved = 0
for i in range(1, 1000):
        energies[i] = int(events_energy_dict[i] / 0.1) #make binning of 100GeV
        if (energies[i] < 5):
            num_moved = num_moved+1
            energies[i] = 40
        if energies[i] > max_val:
            max_val = energies[i]
        if energies[i] < min_val:
            min_val = energies[i]
print ("Num moved: " + str(num_moved))
print("Min: " + str(min_val) + " max: " + str(max_val) + " diff: " + str(max_val - min_val))
energy_bins = max_val - min_val
plt.hist(energies, bins=energy_bins)    
plt.title("Energy distribution")
plt.xlabel("Energy (x 100GeV)")
plt.ylabel("Counts")    
plt.show()


energy_file.close()
#print(lines)
