#!/bin/bash

DIR="$(pwd)"
echo "Current directory: " $DIR

FILE='vgg16_detect_socket_iter_68800.caffemodel'
ID='0B7GupvVgwhysMW90RmJmVjNQUGs'
CHECKSUM=81ad03f85c088839962933442e2ee6da

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
