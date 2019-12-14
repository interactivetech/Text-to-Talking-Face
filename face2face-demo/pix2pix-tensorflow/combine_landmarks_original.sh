# Combine both resized original and landmark images

PYTHON=/root/anaconda3/envs/face2face/bin/python
# INPUT_DIR=/home/face2face-demo/pix2pix-tensorflow/landmarks_resized
INPUT_DIR=/home/face2face-demo/pix2pix-tensorflow/landmarks

# PHOTO_DIR=/home/face2face-demo/pix2pix-tensorflow/original_resized
PHOTO_DIR=/home/face2face-demo/pix2pix-tensorflow/original

OUTPUT_DIR=/home/face2face-demo/pix2pix-tensorflow
$PYTHON tools/process.py \
  --input_dir $INPUT_DIR \
  --b_dir  $PHOTO_DIR \
  --operation combine \
  --output_dir $OUTPUT_DIR/combined