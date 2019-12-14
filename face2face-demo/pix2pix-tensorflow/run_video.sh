# run video
PYTHON=/root/anaconda3/envs/face2face/bin/python
# VIDEO_PATH=/home/face2face-demo/initial_data/angela_merkel_speech.mp4
# VIDEO_PATH=/home/face2face-demo/initial_data/obama_crop3.mp4
# VIDEO_PATH=/home/face2face-demo/initial_data/obama_4000_frames.mp4
# VIDEO_PATH=/home/face2face-demo/initial_data/obama_video_1.mp4
# VIDEO_PATH=/home/face2face-demo/initial_data/Shannon_2_vid_crop_audio.mp4

VIDEO_PATH=/home/face2face-demo/TTTF_ATFL_video/PD_pts_ws_negate_256.mp4
LANDMARK_PATH=/home/face2face-demo/shape_predictor_68_face_landmarks.dat
# TF_MODEL_PATH=/home/face2face-demo/experiments/initial_train/face2face-model-2-epochs/face2face-reduced-model/frozen_model.pb
# TF_MODEL_PATH=/home/face2face-demo/frozen_model.pb
# REDUCED_MODEL_DIR=/home/face2face-demo/experiments/initial_train/face2face-model-9-epochs
# REDUCED_MODEL_DIR=/home/face2face-demo/experiments/obama_face_test/face2face-model-5-epochs
# REDUCED_MODEL_DIR=/home/face2face-demo/experiments/obama_face_test/face2face-model-200-epochs/
# REDUCED_MODEL_DIR=/home/face2face-demo/experiments/obama_face_3997/face2face-model-200-epochs/
# REDUCED_MODEL_DIR=/home/face2face-demo/experiments/obama_face_align_crop/face2face-model-200-epochs/
# REDUCED_MODEL_DIR=/home/face2face-demo/experiments/female_face_exp/face2face-model-8-epochs
REDUCED_MODEL_DIR=/home/face2face-demo/experiments/female_face_exp/face2face-model-200-epochs

TF_MODEL_PATH=$REDUCED_MODEL_DIR/face2face-reduced-model/frozen_model.pb

$PYTHON ../run_video.py \
  --source $VIDEO_PATH --show 0 \
  --landmark-model $LANDMARK_PATH \
  --tf-model $TF_MODEL_PATH
