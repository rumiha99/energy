# A と B のディレクトリのリストをループ
for dir in /home/data/yolo_crop/{A,B,CD}/shiitake/; do
  # ディレクトリに移動
  cd "$dir"
  
  # すべてのファイルを親ディレクトリに移動
  mv * ../
  
  # 親ディレクトリに戻る
  cd ..
  
  # shiitake ディレクトリを削除
  rmdir shiitake
done