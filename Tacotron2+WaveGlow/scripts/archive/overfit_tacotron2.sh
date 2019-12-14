#!/bin/bash

mkdir -p overfit_output

CHECKPOINT="output/checkpoint_Tacotron2_210"

# python -m multiproc train.py -m Tacotron2 -o ./output/ -lr 1e-3 --epochs 1501 -bs 128 --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --log-file ./output/nvlog.json --anneal-steps 500 1000 1500 --anneal-factor 0.1 --amp-run

# Finetune on my_voice.
# --anneal-steps 500 1000 1500 \
python train.py -m Tacotron2 \
	--dataset-path data \
	--training-files data/my_voice/my_voice_train_filelist.txt \
	--validation-files data/my_voice/my_voice_valid_filelist.txt \
	--checkpoint-path $CHECKPOINT \
	-o overfit_output/ \
	-lr 1e-3 \
	--epochs 700 \
	--epochs-per-checkpoint 50 \
	-bs 10 \
	--weight-decay 1e-6 \
	--grad-clip-thresh 1.0 \
	--cudnn-enabled \
	--log-file overfit_output/overfit_tacotron2.nvlog.json \
	--anneal-steps 300 500 700 \
	--anneal-factor 0.1 \
	--amp-run

