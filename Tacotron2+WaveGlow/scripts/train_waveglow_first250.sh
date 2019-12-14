#!/bin/bash -e 

export CUDA_VISIBLE_DEVICES=1

mkdir -p first250_output

python train.py -m WaveGlow \
	-o first250_output/ \
	--training-files filelists/ljs_audio_text_train_first_250.txt \
	-lr 1e-4 \
	--epochs 800 \
	-bs 10 \
	--segment-length  8000 \
	--weight-decay 0 \
	--grad-clip-thresh 65504.0 \
	--epochs-per-checkpoint 50 \
	--cudnn-enabled \
	--cudnn-benchmark \
	--log-file first250_output/waveglow.nvlog.json \
	--amp-run

