#!/bin/bash
# Usage:
# ./experiments/scripts/hip_scratch_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is hip.
#
# Example:
# ./experiments/scripts/hip_scratch_end2end.sh 0 VGG16 hip

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
  hip)
    TRAIN_IMDB="hip_train"
    TEST_IMDB="hip_test"
    PT_DIR="hip"
    ITERS=70000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/hip_scratch_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/detect_end2end/solver_scratch.prototxt \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/hip_end2end.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

#time ./tools/test_net.py --gpu ${GPU_ID} \
#  --def models/${PT_DIR}/${NET}/hip_end2end/test.prototxt \
#  --net ${NET_FINAL} \
#  --imdb ${TEST_IMDB} \
#  --cfg experiments/cfgs/hip_end2end.yml \
#  ${EXTRA_ARGS}
