
source ~/.bashrc
echo $1

runnumber=$1
#python 3Dsquarization.py -i /gpfs0/fact/homekeeping/etienne/WAVEFORMS/MidE/gamma_midE_run${runnumber}.hdf5 -o /gpfs0/fact/homekeeping/etienne/SQUASHFORMS/midE/gamma_midE_run${runnumber}.hdf5
#python 3Dsquarization.py -i /gpfs0/fact/homekeeping/etienne/WAVEFORMS/LowE/gamma_lowE_run${runnumber}.hdf5 -o /gpfs0/fact/homekeeping/etienne/SQUASHFORMS/lowE/gamma_lowE_run${runnumber}.hdf5
python 3Dsquarization.py -i /gpfs0/fact/homekeeping/etienne/WAVEFORMS/HighE/gamma_highE_run${runnumber}.hdf5 -o /gpfs0/fact/homekeeping/etienne/SQUASHFORMS/highE/gamma_highE_run${runnumber}.hdf5
