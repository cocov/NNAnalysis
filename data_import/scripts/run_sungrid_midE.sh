
source ~/.bashrc
echo $1
echo $2
python eventio_to_hdf5_waveforms.py -i $1 -o $2 -e 2
