#!/bin/bash -l

#SBATCH --job-name=mini_lowE_small
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --mail-user=etienne.lyard@unige.ch

export CRAY_CUDA_MPS=0 
export THEANO_FLAGS='cuda.root=$CRAY_CUDATOOLKIT_DIR,device=gpu,floatX=float32'

module load daint-gpu
module load Theano/0.8.2-CrayGNU-2016.11-Python-3.5.2
module load TensorFlow/1.0.0-CrayGNU-2016.11-cuda-8.0-Python-3.5.2

export KERAS_BACKEND=tensorflow

output=$1

if [ "$output" == "" ]; then
	output="output_lowE_small_training4.txt"
fi

cd /scratch/snx3000/lyard/NNAnalysis-master

srun -u -o /scratch/snx3000/lyard/results/$output python convolution_squashed.py -t /scratch/snx3000/lyard/datasets/50kInterleavedSquash.hdf5 -v /scratch/snx3000/lyard/datasets/lowE_validation_60kEvts.hdf5 -s /scratch/snx3000/lyard/models/miniception_lowE_small_training_epoch40.hdf5 -m /scratch/snx3000/lyard/models/miniception_lowE_small_training_epoch60.hdf5


