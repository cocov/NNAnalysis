#!/bin/bash
# added by Anaconda3 4.4.0 installer
export PATH="/home/isdc/lyard/anaconda3/bin:$PATH"

source activate ctapipe
cd ~/ctasoft/ctapipe
make develop
cd /scratch/etienne/NNAnalysis-master/data_import

python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/gamma_20deg_0deg_run17710___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/gamma_20_0_17710_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/gamma_20deg_0deg_run17711___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/gamma_20_0_17711_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/gamma_20deg_0deg_run17712___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/gamma_20_0_17712_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/gamma_20deg_0deg_run17713___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/gamma_20_0_17713_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/gamma_20deg_0deg_run17714___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/gamma_20_0_17714_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/gamma_20deg_0deg_run17715___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/gamma_20_0_17715_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/gamma_20deg_0deg_run17716___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/gamma_20_0_17716_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/gamma_20deg_0deg_run17717___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/gamma_20_0_17717_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/gamma_20deg_0deg_run17718___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/gamma_20_0_17718_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/gamma_20deg_0deg_run17719___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/gamma_20_0_17719_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/proton_20deg_0deg_run17855___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/proton_20_0_17855_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/proton_20deg_0deg_run17856___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/proton_20_0_17856_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/proton_20deg_0deg_run17857___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/proton_20_0_17857_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/proton_20deg_0deg_run17858___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/proton_20_0_17858_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/proton_20deg_0deg_run17859___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/proton_20_0_17859_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/proton_20deg_0deg_run17860___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/proton_20_0_17860_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/proton_20deg_0deg_run17861___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/proton_20_0_17861_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/proton_20deg_0deg_run17862___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/proton_20_0_17862_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/proton_20deg_0deg_run17863___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/proton_20_0_17863_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/proton_20deg_0deg_run17864___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/proton_20_0_17864_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/proton_20deg_0deg_run17865___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/proton_20_0_17865_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/proton_20deg_0deg_run17866___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/proton_20_0_17866_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/proton_20deg_0deg_run17867___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/proton_20_0_17867_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/proton_20deg_0deg_run17868___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/proton_20_0_17868_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/proton_20deg_0deg_run17869___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/proton_20_0_17869_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/proton_20deg_0deg_run17870___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/proton_20_0_17870_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/proton_20deg_0deg_run17871___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/proton_20_0_17871_lowE.hdf5 -e 1
python eventio_to_hdf5_one_tel.py -i /scratch/etienne/MC/proton_20deg_0deg_run17872___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel -o /scratch/etienne/HDF5/proton_20_0_17872_lowE.hdf5 -e 1
