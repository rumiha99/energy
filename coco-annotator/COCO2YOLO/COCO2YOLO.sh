#!/bin/bash

input_path="../../coco-annotator/datasets/test/shiitake-zoom-test.json"
output_path="../../coco-annotator/datasets/test/"
program_path="COCO2YOLO.py"

python3 "$program_path" -j "$input_path" -o "$output_path"
 
# ディレクトリの作成
cd "$output_path"
mkdir -p labels images

 
# .txtファイルをlabelsディレクトリに移動
for file in *.txt; do
  if [ -e "$file" ]; then
    mv "$file" labels/
  fi
done
 
# .jpegファイルをimagesディレクトリに移動
for file in *.JPEG; do
  if [ -e "$file" ]; then
    mv "$file" images/
  fi
done
 
echo "Files have been organized into labels and images directories."