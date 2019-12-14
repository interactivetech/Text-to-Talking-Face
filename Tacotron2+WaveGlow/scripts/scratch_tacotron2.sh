#!/bin/bash

mkdir -p scratch_output

python train.py -m Tacotron2 \
	-o scratch_output/ \
	--dataset-path data \
	--training-files data/my_voice/my_voice_train_filelist.txt \
	--validation-files data/my_voice/my_voice_valid_filelist.txt \
	-lr 1e-3 \
	--epochs 600 \
	--epochs-per-checkpoint 50 \
	-bs 10 \
	--weight-decay 1e-6 \
	--grad-clip-thresh 1.0 \
	--cudnn-enabled \
	--log-file scratch_output/tacotron2.nvlog.json \
	--anneal-steps 200 350 500 \
	--anneal-factor 0.1 \
	--amp-run

