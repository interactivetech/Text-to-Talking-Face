PYTHON=/root/anaconda3/envs/face2face/bin/python
# MODEL_DIR=/home/face2face-demo/experiments/initial_train/face2face-model-2-epochs
# MODEL_DIR=/home/face2face-demo/experiments/initial_train/face2face-model-9-epochs
# MODEL_DIR=/home/face2face-demo/experiments/obama_face_test/face2face-model-5-epochs
# MODEL_DIR=/home/face2face-demo/experiments/obama_face_test/face2face-model-200-epochs
# MODEL_DIR=/home/face2face-demo/experiments/obama_face_3997/face2face-model-200-epochs
# MODEL_DIR=/home/face2face-demo/experiments/obama_face_align_crop/face2face-model-200-epochs

# MODEL_DIR=/home/face2face-demo/experiments/obama_face_align_crop/face2face-model-200-epochs
# MODEL_DIR=/home/face2face-demo/experiments/female_face_exp/face2face-model-8-epochs
MODEL_DIR=/home/face2face-demo/experiments/female_face_exp/face2face-model-200-epochs

$PYTHON ../reduce_model.py \
  --model-input $MODEL_DIR \
  --model-output $MODEL_DIR/face2face-reduced-model

#   Freeze model
  $PYTHON ../freeze_model.py \
    --model-folder $MODEL_DIR/face2face-reduced-model
