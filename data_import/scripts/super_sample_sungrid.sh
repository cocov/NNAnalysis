#!/bin/bash

# added by Anaconda3 4.4.0 installer
export PATH="/home/isdc/lyard/anaconda3/bin:$PATH"

source activate ctapipe
cd ~/ctasoft/ctapipe
make develop
cd /scratch/etienne/NNAnalysis-master/data_import

source_file=$1

python super_sample_no_clean.py -i $source_file".hdf5" -o $source_file"_square.hdf5"



























