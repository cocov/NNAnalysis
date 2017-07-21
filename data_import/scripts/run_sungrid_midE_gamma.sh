
source ~/.bashrc
echo $1

runnumber=$1
python eventio_to_hdf5_waveforms.py -i /gpfs0/fact/homekeeping/etienne/MC/gamma_20deg_0deg_run${runnumber}___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /gpfs0/fact/homekeeping/etienne/WAVEFORMS/MidE/gamma_midE_run${runnumber}.hdf5 -e 2
