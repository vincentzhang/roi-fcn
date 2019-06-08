#!/bin/bash

DIR="$(pwd)"
echo "Current directory: " $DIR

FILE='VGG16.v2.fcn-surgery-all.caffemodel'
ID='1Li-Iky72GKb2beWqHBiMrmirEr-SmTc9'
CHECKSUM=523de2cdb7be6470dad3c9ff8ff6b057

if [ -f "../imagenet_models/$FILE" ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum "../imagenet_models/$FILE" | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat "../imagenet_models/$FILE" | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading ROI-FCN demo models ..."
python google_drive.py $ID "../imagenet_models/$FILE"
echo "Done. Please run this command again to verify that checksum = $CHECKSUM."
