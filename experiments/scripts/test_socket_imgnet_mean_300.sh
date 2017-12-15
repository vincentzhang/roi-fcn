# example: ./experiments/scripts/test_socket_imgnet_mean.sh train/test partial/all {NUM_IMG_PER_EPOCH} {START_EPOCH} {END_EPOCH}
# example: ./experiments/scripts/test_socket_imgnet_mean_300.sh test all 4586 1 16
# example: ./experiments/scripts/test_socket_imgnet_mean.sh test partial, partial means only use slices with positive labels
SPLIT=$1
SUBSET=$2
NUM_IMG_PER_EPOCH=$3 #4586
START_EPOCH=$4
END_EPOCH=$5

GPU_ID=0
NET=VGG16
DATASET=socket
TEST_IMDB="socket_${SPLIT}_${SUBSET}"
PT_DIR="socket"

LOG="experiments/logs/eval_socket_end2end_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
#exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

for i in $(seq $START_EPOCH $END_EPOCH); 
  do 
  echo "Processing epoch: $i"
  ITER=$(($i*$NUM_IMG_PER_EPOCH))
  echo "ITER: ${ITER}"
  #NET_FINAL="output/socket_end2end/socket_train/vgg16_detect_socket_iter_68800.caffemodel"
  #NET_FINAL="output/socket_end2end/socket_train/vgg16_detect_socket_m_300_1e-10_iter_${ITER}.caffemodel"
  NET_FINAL="output/socket_end2end/socket_train/vgg16_detect_socket_300_1e-10_iter_${ITER}.caffemodel"
  echo "Network Model: ${NET_FINAL}"

time ./tools/test_net_seg.py --gpu ${GPU_ID} \
   --def models/${PT_DIR}/${NET}/detect_end2end/test.prototxt \
   --net ${NET_FINAL} \
   --imdb ${TEST_IMDB} \
   --cfg experiments/cfgs/socket_imgnet_mean_end2end_300.yml \
   --suffix ${ITER}
done

