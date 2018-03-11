#!/usr/bin/sh

SEG_DIR="data/segmentation"
mkdir -p $SEG_DIR

(cd $SEG_DIR && \
  curl -L "https://drive.google.com/uc?id=0B_4LoEXGO1TwZEJyMFhuelhSME0" \
       -o "drn_d_22_cityscapes.pth"
)

