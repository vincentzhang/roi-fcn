GPU_ID=1
NET=VGG16
DATASET=pascal

#TEST_IMDB="voc_2007_test_Segmentation"
TEST_IMDB="voc_2011_seg11valid_Segmentation"
PT_DIR="pascal_voc"

NET_FINAL="data/pascal/vgg16_detect_iter_70000.caffemodel"
LOG="experiments/logs/eval_pascal_end2end_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net_seg.py --gpu ${GPU_ID} \
   --def models/${PT_DIR}/${NET}/detect_end2end/test.prototxt \
   --net ${NET_FINAL} \
   --imdb ${TEST_IMDB} \
   --cfg experiments/cfgs/detect_end2end.yml \
   --suffix 70000

