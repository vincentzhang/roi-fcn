# example: ./experiments/scripts/test_socket_net.sh train/test partial/all {NUM_IMG_PER_EPOCH} {START_EPOCH} {END_EPOCH}
# example: ./experiments/scripts/test_socket_net.sh train all 4586 1 16
# example: ./experiments/scripts/test_socket_net.sh test partial, partial means only use slices with positive labels
SPLIT=$1
SUBSET=$2
NUM_IMG_PER_EPOCH=$3 #4586
START_EPOCH=$4
END_EPOCH=$5

GPU_ID=$6
NET=VGG16
DATASET=socket
TEST_IMDB="socket_${SPLIT}_${SUBSET}"
PT_DIR="socket"

#ITERATION_LIST=(4600 9200 13800 18400 23000 27600 32200 36600 41200 45800 50400 55000 59600 64200 68800)
#ITERATION_LIST=(50400 55000 59600 64200 68800)
LOG="experiments/logs/eval_socket_end2end_dice_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

ITER=68800
echo "ITER: ${ITER}"
NET_FINAL="output/socket_end2end/socket_train/vgg16_detect_socket_iter_${ITER}.caffemodel"
echo "Network Model: ${NET_FINAL}"

time ./tools/test_net_seg.py --gpu ${GPU_ID} \
   --def models/${PT_DIR}/${NET}/detect_end2end/test.prototxt \
   --net ${NET_FINAL} \
   --imdb ${TEST_IMDB} \
   --cfg experiments/cfgs/socket_imgnet_mean_end2end.yml \
   --suffix "RPN_${ITER}_DICE"

