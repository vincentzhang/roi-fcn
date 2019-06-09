#!/bin/bash

DIR="$(pwd)"
echo "Current directory: " $DIR

FILE='vgg16_acce_fg50_1e-04_iter_42960.caffemodel'
ID='1BqSKNKfDjRS8xfSqObqGXExcsF_r42ar'
CHECKSUM=ebb74f273989d40435abfe62edda2f1a

if [ -f "../$FILE" ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum "../$FILE" | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat "../$FILE" | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading ROI-FCN demo models ..."
python google_drive.py $ID "../$FILE"
echo "Done. Please run this command again to verify that checksum = $CHECKSUM."
