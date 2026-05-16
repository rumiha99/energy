#!/bin/bash

# --- ディレクトリパスの設定 ---
CROP_DIR="/home/data/1124_keijotest/crop/C"
MASK_DIR="/home/data/1124_keijotest/mask/C"
COMBINED_DIR="/home/data/1124_keijotest/combined/C"

# --- 削除処理の実行 ---

echo "--- 削除対象の検索と処理を開始します ---"

# maskディレクトリ内の全ファイルをループ処理
# maskファイルは全てのファイルが存在すると仮定して、比較の基準とします。
find "$MASK_DIR" -maxdepth 1 -type f | while read MASK_PATH; do
    
    # maskファイル名のみを取得 (例: IMG_6560_mask.png)
    MASK_FILENAME=$(basename "$MASK_PATH")
    
    # 共通のベース名を取得 (例: IMG_6560)
    # _mask.png を除去する
    BASE_NAME=$(echo "$MASK_FILENAME" | sed -E 's/(_mask)?\.(png|jpg|jpeg)$//')

    # cropディレクトリでの対応ファイル名を作成 (例: IMG_6560_crop.png)
    # 拡張子がpngであると仮定します
    CROP_TARGET="${CROP_DIR}/${BASE_NAME}_crop.png" 

    # cropディレクトリにファイルが残っているか確認（この条件が重要）
    if [ -f "$CROP_TARGET" ]; then
        
        # cropにファイルが残っている場合 (削除しないファイル)
        echo "✅ 残存ファイル: $BASE_NAME - cropにも存在するためスキップ"
        
    else
        
        # cropにファイルが存在しない場合 (ユーザーによって削除されたファイル)
        echo "🔥 削除対象: $BASE_NAME - cropにファイルが存在しません"
        
        # --- maskディレクトリからファイルを削除 ---
        MASK_TARGET="${MASK_DIR}/${MASK_FILENAME}" # MASK_PATHと同じ
        
        if [ -f "$MASK_TARGET" ]; then
            # 実際に削除 (コメントアウトを外すと実行されます)
            rm "$MASK_TARGET"
            echo "   - maskから削除: $MASK_TARGET"
        fi
        
        # --- combinedディレクトリから同名のファイルを削除 ---
        # ターゲットファイル名: ${BASE_NAME}_combined.png
        COMBINED_TARGET="${COMBINED_DIR}/${BASE_NAME}_combined.png"
        
        if [ -f "$COMBINED_TARGET" ]; then
            # 実際に削除 (コメントアウトを外すと実行されます)
            rm "$COMBINED_TARGET"
            echo "   - combinedから削除: $COMBINED_TARGET"
        fi
        
    fi
    
done

echo "--- 処理完了 ---"