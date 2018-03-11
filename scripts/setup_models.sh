#!/bin/sh

# Run from project root
(cd src/thirdparty/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
