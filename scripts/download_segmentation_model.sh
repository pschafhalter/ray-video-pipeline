#!/usr/bin/sh

SEG_DIR="data/segmentation"
mkdir -p $SEG_DIR

(cd $SEG_DIR && \
  curl -L "https://drive.google.com/uc?id=1K3BPYIpcB0qqWPf_anleTta-BDM92KDX" \
       -o "drn_d_22_bdd_v1.pth"
)
