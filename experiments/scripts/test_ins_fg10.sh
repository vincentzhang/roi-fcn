# example: ./experiments/scripts/test_socket_scratch_n.sh train/test partial/all {NUM_IMG_PER_EPOCH} {START_EPOCH} {END_EPOCH}
# example: ./experiments/scripts/test_ins_scratch_n_fg50_e-4.sh train all 4586 1 16
# example: ./experiments/scripts/test_ins_fg10 test partial, partial means only use slices with positive labels
export PYTHONUNBUFFERED="True"

SPLIT=$1
SUBSET=$2
NUM_IMG_PER_EPOCH=$3 #225
START_EPOCH=$4
END_EPOCH=$5

GPU_ID=$6
NET=VGG16
DATASET=ins
TEST_IMDB="ins_${SPLIT}_${SUBSET}"
PT_DIR="ins"


LOG="experiments/logs/eval_ins_fg10_end2end_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

for i in $(seq $START_EPOCH $END_EPOCH); 
  do 
  echo "Processing epoch: $i"
  ITER=$(($i*$NUM_IMG_PER_EPOCH))
  echo "ITER: ${ITER}"
  NET_FINAL="output/ins_end2end/ins_train/vgg16_detect_ins_fg10_1e-10_iter_${ITER}.caffemodel"
  echo "Network Model: ${NET_FINAL}"

time ./tools/test_net_seg.py --gpu ${GPU_ID} \
   --def models/${PT_DIR}/${NET}/detect_end2end/test.prototxt \
   --net ${NET_FINAL} \
   --imdb ${TEST_IMDB} \
   --cfg experiments/cfgs/ins_fg10_end2end.yml \
   --suffix ${ITER}
done

