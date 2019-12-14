#!/bin/bash  
PYTHON=/root/anaconda3/envs/audio-landmarks-python3.6/bin/python
TRAIN_PY=/home/Talking-Face-Landmarks-from-Speech/train.py 
# tail $FEATURE_EXTRACTOR_DIR/featureExtractor.py

PATH_TO_H5=/home/Talking-Face-Landmarks-from-Speech/experiments/obama-1-finetuning/train_dataset/obama-1-dataset.hdf5
OUTPUT_FOLDER=/home/Talking-Face-Landmarks-from-Speech/experiments/obama-1-finetuning/results/

$PYTHON $TRAIN_PY -i $PATH_TO_H5 \
-u 512 -d 1 \
-c 5 -o $OUTPUT_FOLDER -e 1

#  $FEATURE_EXTRACTOR_PY -vp $PATH_TO_VIDEO_FILES -sp $PATH_TO_SHAPE_PREDICTOR -o $OUTPUT_FILE_DIR$OUTPUT_NAME.hdf5