#!/bin/bash
#JSON2YOLOのあとに実施

# 共通部分を変数として定義
BASE_DIR="0728_segmet"
#testとvalにしたい数の合計(test + val)
NUM_FILES_TO_MOVE=100

# フォルダのパスを定義
LABELS_DIR="/home/YOLO/$BASE_DIR/datasets/labels"
IMAGES_DIR="/home/YOLO/$BASE_DIR/datasets/images"

# 必要なフォルダを作成
echo "Creating directories..."
mkdir -p "$LABELS_DIR/val"
mkdir -p "$LABELS_DIR/test"
mkdir -p "$IMAGES_DIR/$BASE_DIR"
mkdir -p "$IMAGES_DIR/val"
mkdir -p "$IMAGES_DIR/test"

# idofolder2folder.pyのパラメータ
SOURCE_FOLDER="$IMAGES_DIR/$BASE_DIR"
DESTINATION_FOLDER1="$IMAGES_DIR/test"
DESTINATION_FOLDER2="$IMAGES_DIR/val"

# ido_samename.pyのパラメータ
SOURCE_FOLDER_SAMENAME="$IMAGES_DIR/val"
COMPARE_FOLDER="$LABELS_DIR/$BASE_DIR"
DESTINATION_FOLDER_SAMENAME="$LABELS_DIR/val"

# ido_samename2.pyのパラメータ
SOURCE_FOLDER_SAMENAME2="$IMAGES_DIR/test"
COMPARE_FOLDER2="$LABELS_DIR/$BASE_DIR"
DESTINATION_FOLDER_SAMENAME2="$LABELS_DIR/test"

# cp_allfile.pyのパラメータ
COPY_SOURCE_FOLDER="/home/coco-annotator/datasets/$BASE_DIR"
COPY_DESTINATION_FOLDER="$IMAGES_DIR/$BASE_DIR"

# cp_allfile.pyを実行
python3 /home/src/before_coco2yolo/cp_allfile.py "$COPY_SOURCE_FOLDER" "$COPY_DESTINATION_FOLDER"

# idofolder2folder.pyを実行
python3 /home/src/before_coco2yolo/idofolder2folder.py "$SOURCE_FOLDER" "$DESTINATION_FOLDER1" "$DESTINATION_FOLDER2" "$NUM_FILES_TO_MOVE"

# ido_samename.pyを実行
python3 /home/src/before_coco2yolo/ido_samename.py "$SOURCE_FOLDER_SAMENAME" "$COMPARE_FOLDER" "$DESTINATION_FOLDER_SAMENAME"

# ido_samename2.pyを実行
python3 /home/src/before_coco2yolo/ido_samename2.py "$SOURCE_FOLDER_SAMENAME2" "$COMPARE_FOLDER2" "$DESTINATION_FOLDER_SAMENAME2"


