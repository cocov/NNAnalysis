
source ~/.bashrc
echo $1

runnumber=$1
python 3Dsquarization.py -i /gpfs0/fact/homekeeping/etienne/WAVEFORMS/MidE/proton_midE_run${runnumber}.hdf5 -o /gpfs0/fact/homekeeping/etienne/SQUASHFORMS/midE/proton_midE_run${runnumber}.hdf5
python 3Dsquarization.py -i /gpfs0/fact/homekeeping/etienne/WAVEFORMS/LowE/proton_lowE_run${runnumber}.hdf5 -o /gpfs0/fact/homekeeping/etienne/SQUASHFORMS/lowE/proton_lowE_run${runnumber}.hdf5
python 3Dsquarization.py -i /gpfs0/fact/homekeeping/etienne/WAVEFORMS/HighE/proton_highE_run${runnumber}.hdf5 -o /gpfs0/fact/homekeeping/etienne/SQUASHFORMS/highE/proton_highE_run${runnumber}.hdf5
