import os
import shutil

def copy_all_files(source_folder, destination_folder):
    # 移動元フォルダからファイルのリストを取得
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # ファイルを移動先フォルダにコピー
    for file in files:
        src_path = os.path.join(source_folder, file)
        dst_path = os.path.join(destination_folder, file)
        shutil.copy(src_path, dst_path)
        print(f'Copied: {file}')

if __name__ == "__main__":
    import sys
    source_folder = sys.argv[1]
    destination_folder = sys.argv[2]
    copy_all_files(source_folder, destination_folder)