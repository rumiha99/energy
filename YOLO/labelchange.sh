#!/bin/bash
BASE_DIR="0728_segmet"
# 処理対象のディレクトリリスト
directories=(
    "/home/YOLO/$BASE_DIR/datasets/labels/$BASE_DIR"
    "/home/YOLO/$BASE_DIR/datasets/labels/test"
    "/home/YOLO/$BASE_DIR/datasets/labels/val"
)

# 各ディレクトリごとに処理を行う
for directory in "${directories[@]}"; do
    for file in "$directory"/*.txt; do
        # ファイルが存在するか確認
        [ -e "$file" ] || continue
        
        # 各行の最初の数値が 4 なら 0 に置き換える
        awk '{if ($1 == "3") $1 = "0"; print}' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    done
done
