#!/usr/bin/sh

FUSION_DIR="data/fusion"
mkdir -p $FUSION_DIR

(cd $FUSION_DIR && \
  wget "http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy" &&
  wget "https://rise-driving.s3.amazonaws.com/train/alexnet_quantized_angles_5_2/model.ckpt-990.data-00000-of-00001" &&
  wget "https://rise-driving.s3.amazonaws.com/train/alexnet_quantized_angles_5_2/model.ckpt-990.index" &&
  wget "https://rise-driving.s3.amazonaws.com/train/alexnet_quantized_angles_5_2/model.ckpt-990.meta"
)
