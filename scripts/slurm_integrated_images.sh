#!/bin/bash -l
#SBATCH --job-name=integrated_images
#SBATCH --time=05:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --mail-user=etienne.lyard@unige.ch

export CRAY_CUDA_MPS=1 
export THEANO_FLAGS='cuda.root=$CRAY_CUDATOOLKIT_DIR,device=gpu,floatX=float32'

module load daint-gpu
module load Theano/0.8.2-CrayGNU-2016.11-Python-3.5.2

cd /users/lyard/NNAnalysis-master
srun -u -o ~/output.txt python integrated_images.py -t data_import/training_10000evts_square.hdf5 -v data_import/validation_1000evts_square.hdf5 
