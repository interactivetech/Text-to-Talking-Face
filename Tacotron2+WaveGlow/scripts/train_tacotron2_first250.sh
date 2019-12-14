#!/bin/bash

mkdir -p first250_output

python train.py -m Tacotron2 \
	-o first250_output/ \
	--training-files filelists/ljs_audio_text_train_first_250.txt \
	-lr 1e-3 \
	--epochs 600 \
	--epochs-per-checkpoint 50 \
	-bs 10 \
	--weight-decay 1e-6 \
	--grad-clip-thresh 1.0 \
	--cudnn-enabled \
	--log-file first250_output/tacotron2.nvlog.json \
	--anneal-steps 200 350 500 \
	--anneal-factor 0.1 \
	--amp-run

