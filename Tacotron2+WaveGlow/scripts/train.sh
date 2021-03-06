#!/bin/bash -e

MODEL=
OUTPUT_DIR=
EPOCHS_PER_CKPT=2

# Default was 1500 for Taco, 1000 for Wave:
NUM_EPOCHS=

# Default was 80 for Taco, 10 for Wave:
BATCH_SIZE=

DATA="./"
TRAINING_FILES="filelists/ljs_audio_text_train_filelist.txt"
VALIDATION_FILES="filelists/ljs_audio_text_val_filelist.txt"

for arg in "$@"; do
    case $arg in
        -m=*|--model=*)
        MODEL="${arg#*=}"
        shift
        ;;
        -o=*|--output-dir=*)
        OUTPUT_DIR="${arg#*=}"
        shift
        ;;
        -e=*|--epochs-per-ckpt=*)
        EPOCHS_PER_CKPT="${arg#*=}"
        shift
        ;;
        --num-epochs=*)
        NUM_EPOCHS="${arg#*=}"
        shift
        ;;
        -d=*|--data=*)
        DATA="${arg#*=}"
        shift
        ;;
        -b=*|--batch=*)
        BATCH_SIZE="${arg#*=}"
        shift
        ;;
        --train=*)
        TRAINING_FILES="${arg#*=}"
        shift
        ;;
        --valid=*)
        VALIDATION_FILES="${arg#*=}"
        shift
        ;;
        -w)
        ORGANIZE_WAVEGLOW_CKPTS=1
        shift
        ;;
        --help)
        echo "ur dumb"
        exit
        ;;
        *)
        echo "Unknown option " $i
        echo "ur dumb"
        exit 1
        ;;
    esac
done

if [[ -z "${MODEL}" ]]; then
    echo "Must specify -m"
    exit 1
elif [[ -z "${OUTPUT_DIR}" ]]; then 
    echo "Must specify -o"
    exit 1
elif [[ -z "${BATCH_SIZE}" ]]; then 
    echo "Must specify -b|--batch"
    exit 1
fi

mkdir -p $OUTPUT_DIR

if [[ "${MODEL}" == "Tacotron2" ]]; then
    echo "Running train.py for Tacotron2..."
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    nohup python -m multiproc train.py -m Tacotron2 \
        -o "$OUTPUT_DIR/" \
        --dataset-path ${DATA} \
        --dist-url="tcp://localhost:23455" \
        --training-files ${TRAINING_FILES} \
        --validation-files ${VALIDATION_FILES} \
        -lr 1e-3 \
        --epochs ${NUM_EPOCHS} \
        --epochs-per-checkpoint ${EPOCHS_PER_CKPT} \
        -bs $BATCH_SIZE \
        --weight-decay 1e-6 \
        --grad-clip-thresh 1.0 \
        --cudnn-enabled \
        --log-file $OUTPUT_DIR/tacotron2.nvlog.json \
        --anneal-steps 500 1000 1500 \
        --anneal-factor 0.1 \
        --amp-run > $OUTPUT_DIR/tacotron2.out &
    echo $! > $OUTPUT_DIR/tacotron2.pid.txt
    echo "PID saved in $OUTPUT_DIR/tacotron2.pid.txt: $(cat "$OUTPUT_DIR/tacotron2.pid.txt")"
elif [[ "${MODEL}" == "WaveGlow" ]]; then
    echo "Running train.py for WaveGlow..."
    export CUDA_VISIBLE_DEVICES="4,5,6,7"
    # export CUDA_VISIBLE_DEVICES="0,1,2,3"
    nohup python -m multiproc train.py -m WaveGlow \
        -o "$OUTPUT_DIR/" \
        --dist-url="tcp://localhost:23458" \
        --dataset-path ${DATA} \
        --training-files ${TRAINING_FILES} \
        --validation-files ${VALIDATION_FILES} \
        -lr 1e-4 \
        --epochs ${NUM_EPOCHS} \
        -bs $BATCH_SIZE \
        --segment-length  8000 \
        --weight-decay 0 \
        --grad-clip-thresh 65504.0 \
        --epochs-per-checkpoint ${EPOCHS_PER_CKPT} \
        --cudnn-enabled \
        --cudnn-benchmark \
        --log-file $OUTPUT_DIR/waveglow.nvlog.json \
        --amp-run > $OUTPUT_DIR/waveglow.out &
    echo $! > $OUTPUT_DIR/waveglow.pid.txt
    echo "PID saved in $OUTPUT_DIR/waveglow.pid.txt: $(cat "$OUTPUT_DIR/waveglow.pid.txt")"
fi


