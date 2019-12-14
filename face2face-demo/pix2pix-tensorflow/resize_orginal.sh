PYTHON=/root/anaconda3/envs/face2face/bin/python
INPUT_DIR=/home/face2face-demo/pix2pix-tensorflow/original
OUTPUT_DIR=/home/face2face-demo/pix2pix-tensorflow
$PYTHON tools/process.py \
  --input_dir $INPUT_DIR \
  --operation resize \
  --size 1024 \
  --output_dir $OUTPUT_DIR/original_resized