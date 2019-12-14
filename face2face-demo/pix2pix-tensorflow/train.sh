PYTHON=/root/anaconda3/envs/face2face/bin/python
# EXP_DIR=/home/face2face-demo/experiments/obama_face_test
# EXP_DIR=/home/face2face-demo/experiments/obama_face_3997
# EXP_DIR=/home/face2face-demo/experiments/obama_face_align_crop
EXP_DIR=/home/face2face-demo/experiments/female_face_exp
# EPOCHS=200
# EPOCHS=200
EPOCHS=200
EXP_NAME=face2face-model-$EPOCHS-epochs
# INPUT_DIR=/home/face2face-demo/pix2pix-tensorflow/combined/train
INPUT_DIR=/home/face2face-demo/pix2pix-tensorflow/combined

python pix2pix.py \
  --mode train \
  --output_dir $EXP_DIR/$EXP_NAME \
  --max_epochs $EPOCHS \
  --input_dir $INPUT_DIR \
  --scale_size 256 \
  --which_direction AtoB