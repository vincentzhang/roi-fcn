# example: ./experiments/scripts/test_hiph5_net.sh train all {NUM_IMG_PER_EPOCH} {START_EPOCH} {END_EPOCH}
# example: ./experiments/scripts/test_hiph5_net.sh train all 2944 1 16
# example: ./experiments/scripts/test_hiph5_net.sh test partial, partial means only use slices with positive labels
SPLIT=$1
SUBSET=$2
GPU_ID=1
NET=VGG16
DATASET=hiph5

#TEST_IMDB="hiph5_test"
TEST_IMDB="hiph5_${SPLIT}_${SUBSET}"
PT_DIR="hiph5"

START_EPOCH=$4
END_EPOCH=$5
NUM_IMG_PER_EPOCH=$3 #2944

#NET_FINAL="output/hip_end2end/hiph5_train/vgg16_detect_hiph5_iter_50000.caffemodel"
LOG="experiments/logs/eval_hiph5_end2end_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


for i in $(seq $START_EPOCH $END_EPOCH); 
  do 
  echo "Processing epoch: $i"
  ITER=$(($i*$NUM_IMG_PER_EPOCH))
  echo "ITER: ${ITER}"
  NET_FINAL="output/hiph5_end2end/hiph5_train/vgg16_detect_hiph5_iter_${ITER}.caffemodel"
  echo "Network Model: ${NET_FINAL}"
  time ./tools/test_net_seg.py --gpu ${GPU_ID} \
     --def models/${PT_DIR}/${NET}/detect_end2end/test.prototxt \
     --net ${NET_FINAL} \
     --imdb ${TEST_IMDB} \
     --cfg experiments/cfgs/hiph5_end2end.yml \
     --suffix ${ITER}
done
