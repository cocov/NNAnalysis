#!/bin/bash -l

#SBATCH --job-name=integrated_images
#SBATCH --time=23:59:00
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
optimizer=$2
loss=$3

if [ "$output" == "" ]; then
	output="output_inception.txt"
fi

if [ "$optimizer" == "" ]; then
	optimizer="sgd"
fi

if [ "$loss" == "" ]; then
	loss="mse"
fi

cd /scratch/snx3000/lyard/NNAnalysis-master
srun -u -o /scratch/snx3000/lyard/$output python inceptionV3.py -o $optimizer -l $loss -t 50k_evts_3_chans_square.hdf5 -v 1k_evts_3_chans_square.hdf5
