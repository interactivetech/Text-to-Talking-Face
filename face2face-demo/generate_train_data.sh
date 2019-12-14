#/bin/sh
PYTHON=/root/anaconda3/envs/face2face/bin/python
# VIDEO_PATH=/home/face2face-demo/initial_data/angela_merkel_speech.mp4
# VIDEO_PATH=/home/face2face-demo/initial_data/obama_crop3.mp4
# VIDEO_PATH=/home/face2face-demo/initial_data/obama_4000_frames.mp4
# VIDEO_PATH=/home/face2face-demo/initial_data/obama_video_1.mp4
VIDEO_PATH=/home/face2face-demo/initial_data/Shannon_2_vid_crop_audio.mp4
LANDMARK_PATH=/home/face2face-demo/shape_predictor_68_face_landmarks.dat
# --num 4000 \

$PYTHON generate_train_data.py \
--file $VIDEO_PATH \
--num 7000 \
--landmark-model $LANDMARK_PATH
