#!/bin/bash
# Usage:
# ./experiments/scripts/ins_scratch_n_fg10_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is hip.
#
# Example:
# ./experiments/scripts/ins_fg10_end2end.sh 0 VGG16 ins\
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  ins)
    TRAIN_IMDB="ins_train_partial"
    TEST_IMDB="ins_test_all"
    PT_DIR="ins"
    ITERS=70000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/ins_fg10_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/detect_end2end/solver_fg10.prototxt \
  --weights data/imagenet_models/VGG16_faster_rcnn_final-surgery-all.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/ins_fg10_end2end.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

