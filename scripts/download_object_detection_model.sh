#!/usr/bin/sh

OD_DIR="data/object_detection"
MODEL="ssd_mobilenet_v1_coco_2017_11_17"
mkdir -p $OD_DIR

(cd $OD_DIR && \
  wget "http://download.tensorflow.org/models/object_detection/$MODEL.tar.gz" && \
  tar -zxvf "$MODEL.tar.gz" "$MODEL/frozen_inference_graph.pb" && \
  rm "$MODEL.tar.gz" # Cleanup
)
