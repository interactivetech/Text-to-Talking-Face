#!/bin/bash  
PYTHON=/root/anaconda3/envs/audio-landmarks-python3.6/bin/python
FEATURE_EXTRACTOR_PY=/home/Talking-Face-Landmarks-from-Speech/featureExtractor.py 
# tail $FEATURE_EXTRACTOR_DIR/featureExtractor.py

PATH_TO_VIDEO_FILES=/home/Talking-Face-Landmarks-from-Speech/obama-1-dataset
PATH_TO_SHAPE_PREDICTOR=/home/Talking-Face-Landmarks-from-Speech/shape_predictor_68_face_landmarks.dat
OUTPUT_FILE_DIR=/home/Talking-Face-Landmarks-from-Speech/experiments/obama-1-finetuning/train_dataset/
OUTPUT_NAME=obama-1-dataset

$PYTHON $FEATURE_EXTRACTOR_PY -vp $PATH_TO_VIDEO_FILES -sp $PATH_TO_SHAPE_PREDICTOR -o $OUTPUT_FILE_DIR$OUTPUT_NAME.hdf5