GPU_ID=0
NET=VGG16
DATASET=socket

TEST_IMDB="socket_test"
PT_DIR="socket"

# Single line version
#NET_FINAL="output/socket_end2end/socket_train/vgg16_detect_hip_iter_70000.caffemodel"
#NET_FINAL="output/socket_end2end/socket_train/vgg16_detect_hip_iter_4000.caffemodel"
# Band version
#NET_FINAL="output/socket_end2end/socket_train/vgg16_detect_socket_iter_1800.caffemodel"
NET_FINAL="output/socket_end2end/socket_train/vgg16_detect_socket_iter_200.caffemodel"
LOG="experiments/logs/eval_socket_end2end_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net_seg.py --gpu ${GPU_ID} \
   --def models/${PT_DIR}/${NET}/detect_end2end/test.prototxt \
   --net ${NET_FINAL} \
   --imdb ${TEST_IMDB} \
   --cfg experiments/cfgs/socket_end2end.yml \
   --suffix 200

