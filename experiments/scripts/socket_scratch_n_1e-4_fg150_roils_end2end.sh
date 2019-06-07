#!/bin/bash
# Usage:
# ./experiments/scripts/socket_scratch_n_fg150_roils_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is socket.
#
# Example:
# ./experiments/scripts/socket_scratch_n_1e-4_fg150_roils_end2end.sh 0 VGG16 socket

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
  socket)
    TRAIN_IMDB="socket_train_partial"
    TEST_IMDB="socket_test_all"
    PT_DIR="socket"
    ITERS=70000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/socket_scratch_n_m_1e-4_fg150_roils_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/detect_end2end/solver_scratch_n_1e-4_fg150_roils.prototxt \
  --weights data/imagenet_models/VGG16.v2.fcn-surgery-all.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/socket_scratch_n_fg150_roils_end2end.yml \
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
