#!/bin/bash -e 

mkdir -p overfit_output
export CUDA_VISIBLE_DEVICES=1

CHECKPOINT="output/checkpoint_WaveGlow_48"

# python -m multiproc train.py -m WaveGlow -o ./output/ -lr 1e-4 --epochs 1001 -bs 10 --segment-length  8000 --weight-decay 0 --grad-clip-thresh 65504.0 --cudnn-enabled --cudnn-benchmark --log-file ./output/nvlog.json --amp-run

python train.py -m WaveGlow \
	-o overfit_output/ \
	--dataset-path data \
	--training-files data/my_voice/my_voice_train_filelist.txt \
	--validation-files data/my_voice/my_voice_train_filelist.txt \
	--checkpoint-path $CHECKPOINT \
	-lr 1e-4 \
	--epochs 800 \
	--epochs-per-checkpoint 50 \
	-bs 10 \
	--segment-length 8000 \
	--weight-decay 0 \
	--grad-clip-thresh 65504.0 \
	--cudnn-enabled \
	--cudnn-benchmark \
	--log-file overfit_output/nvlog.json \
	--amp-run

