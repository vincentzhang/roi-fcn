GPU_ID=0
NET=VGG16
DATASET=hip

TEST_IMDB="hip_test"
PT_DIR="hip"

NET_FINAL="data/vgg16_detect_hip_iter_70000.caffemodel"
LOG="experiments/logs/eval_hip_end2end_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net_seg.py --gpu ${GPU_ID} \
   --def models/${PT_DIR}/${NET}/detect_end2end/test.prototxt \
   --net ${NET_FINAL} \
   --imdb ${TEST_IMDB} \
   --cfg experiments/cfgs/hip_end2end.yml

