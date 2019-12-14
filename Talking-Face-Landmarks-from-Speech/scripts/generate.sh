
PYTHON=/root/anaconda3/envs/audio-landmarks-python3.6/bin/python
# INPUT_AUDIO=/home/Talking-Face-Landmarks-from-Speech/experiments/obama-1-finetuning/ATFL_results/test_audio/16_sec_obama.mp4.wav
INPUT_AUDIO=/home/Talking-Face-Landmarks-from-Speech/TTS-audio/taco_1500_waveglow_950_0.wav
GENERATE_PY=/home/Talking-Face-Landmarks-from-Speech/generate.py 

# EXPERIMENT_DIR=/home/Talking-Face-Landmarks-from-Speech/experiments/obama-1-finetuning/results/exp_ATFL_hid_units=512_delay=1_ctx=5_epochs=1
MODEL_DIR=/home/Talking-Face-Landmarks-from-Speech/experiments/obama-1-finetuning/results/exp_ATFL_hid_units=512_delay=1_ctx=5_epochs=1
MODEL=$MODEL_DIR/talkingFaceModel.h5
# OUTPUT_DIR=$EXPERIMENT_DIR/landmarks_res_256
OUTPUT_DIR=/home/Talking-Face-Landmarks-from-Speech/experiments/obama_train_TTS_voice/gen_voice_res

$PYTHON $GENERATE_PY \
-i  $INPUT_AUDIO \
-m  $MODEL \
-o  $OUTPUT_DIR \
-c 5 -d 1 