#!/bin/bash -e 

mkdir -p scratch_output
export CUDA_VISIBLE_DEVICES=1

python train.py -m WaveGlow \
	-o scratch_output/ \
	--dataset-path data \
	--training-files data/my_voice/my_voice_train_filelist.txt \
	--validation-files data/my_voice/my_voice_train_filelist.txt \
	-lr 1e-4 \
	--epochs 800 \
	--epochs-per-checkpoint 50 \
	-bs 10 \
	--segment-length 8000 \
	--weight-decay 0 \
	--grad-clip-thresh 65504.0 \
	--cudnn-enabled \
	--cudnn-benchmark \
	--log-file scratch_output/waveglow.nvlog.json \
	--amp-run

