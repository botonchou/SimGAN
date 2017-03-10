#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/boton/Dropbox/CMU/gan/deep-learning-utils/

dir=/share/Research/dataset/gan/

python sim-gan.py $dir/SynthEyes_data $dir/MPII_Gaze_Dataset
