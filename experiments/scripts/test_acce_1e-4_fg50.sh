# example: ./experiments/scripts/test_socket_scratch_n.sh train/test partial/all {NUM_IMG_PER_EPOCH} {START_EPOCH} {END_EPOCH}
# example: ./experiments/scripts/test_acce_1e-4_fg50.sh test all 1790 1 27 1
# example: ./experiments/scripts/test_socket_scratch_n_fg.sh test partial, partial means only use slices with positive labels
export PYTHONUNBUFFERED="True"

SPLIT=$1
SUBSET=$2
NUM_IMG_PER_EPOCH=$3 #1790
START_EPOCH=$4
END_EPOCH=$5

GPU_ID=$6
NET=VGG16
DATASET=acce
TEST_IMDB="acce_${SPLIT}_${SUBSET}"
PT_DIR="socket"


LOG="experiments/logs/eval_acce_1e-4_fg50_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

for i in $(seq $START_EPOCH $END_EPOCH); 
  do 
  echo "Processing epoch: $i"
  ITER=$(($i*$NUM_IMG_PER_EPOCH))
  echo "ITER: ${ITER}"
  NET_FINAL="output/acce_end2end/acce_train/vgg16_acce_fg50_1e-04_iter_${ITER}.caffemodel"
  echo "Network Model: ${NET_FINAL}"

    time ./tools/test_net_seg.py --gpu ${GPU_ID} \
       --def models/${PT_DIR}/${NET}/detect_end2end/test.prototxt \
       --net ${NET_FINAL} \
       --imdb ${TEST_IMDB} \
       --cfg experiments/cfgs/acce_fg50.yml \
       --suffix ${ITER}
  done

