#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/boton/Dropbox/CMU/gan/deep-learning-utils/

data_dir=/share/Research/dataset/gan/
exp_dir=/share/Research/gan-exp/eyes/

./sim-gan.py \
  --synthetic-data-dir $data_dir/SynthEyes_data \
  --real-data-dir $data_dir/MPII_Gaze_Dataset \
  --exp-dir $exp_dir \
  --refiner-model-path refiner_model_step_10400.h5 \
  --discriminator-model-path discriminator_model_step_10400.h5
