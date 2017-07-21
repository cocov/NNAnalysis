'''
Created on 7 Jul 2017

@author: lyard
'''

import matplotlib.pyplot as plt
from optparse import OptionParser
import numpy as np

if __name__ == '__main__':
    op = OptionParser()
    op.add_option("-i", "--input", dest="input", help="file to plot", default="", type=str)
    op.add_option("-t", "--threshold", dest="threshold", help="discrimination threshold", default="0.5", type=float)
    
    (options, args) = op.parse_args()
    
    input_filename = options.input
    threshold = options.threshold
    
    input = open(input_filename)
    
    lines = input.readlines()
    
    false_positive = 0

    missed_gamma = 0

    counter = 0
    
    num_bins = 100
    gamma_vals = np.empty(65000)
    hadron_vals = np.empty(65000)
    for i in range(65000):
        gamma_vals[i] = 0
        hadron_vals[i] = 0
    
    num_hadrons = 0
    num_gammas = 0
    for line in lines:
        counter = counter + 1
        bracket = line.find('[')
    
        end_bracket = line.find(']')
    
        numbers = line[bracket+1:end_bracket]
    
        splitted = numbers.split(',')
    
        first_value = float(splitted[0])
        second_value = float(splitted[1])
        remaining_line = line[end_bracket+1:]
    
        bracket = remaining_line.find('[')
        end_bracket = remaining_line.find(']')
    
        numbers = remaining_line[bracket+1:end_bracket]
    
        splitted = numbers.split(' ')
        first_truth = int(splitted[0])
        second_truth = int(splitted[1])
    
    
        if first_value < threshold and first_truth == 1:
            missed_gamma = missed_gamma + 1
        else:
            if first_value > threshold and first_truth == 0:
                false_positive = false_positive + 1

        if first_truth == 1:
            gamma_vals[num_gammas] = first_value
            num_gammas = num_gammas + 1
        else:
            hadron_vals[num_hadrons] = first_value
            num_hadrons = num_hadrons + 1

    accuracy = (float)(counter - (missed_gamma + false_positive))/(float)(counter) 
    print("Total: " + str(counter) + " Threshold: " + str(threshold) + " Missed gammas: " + str(missed_gamma) + " false positive: " + str(false_positive) + " Accuracy: " + str(accuracy*100) + "%")

    exit(0)

    gamma_hist = np.empty(num_gammas)
    hadron_hist = np.empty(num_hadrons)
    for i in range(num_gammas):
        gamma_hist[i] = gamma_vals[i]
    for i in range(num_hadrons):
        hadron_hist[i] = hadron_vals[i]
        
    bins = np.linspace(0,1, 100)
    #plt.hist(bins, num_bins)
    plt.hist(gamma_hist, bins, alpha=0.5, label='gamma')
    plt.hist(hadron_hist, bins, alpha=0.5, label='proton')
    plt.yscale('log')
    plt.title("Values distribution")
    plt.xlabel("Gamma-ness")
    plt.ylabel("Counts")
    plt.legend(loc='best')
    plt.show()