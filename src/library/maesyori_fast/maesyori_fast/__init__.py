import multiprocessing
import time
import os
import sys

# ( . は「同じフォルダ内」の _internal_worker を指す)
from ._internal_worker import process_image_subdirectory

def run(base_dir, model_path, target_folders, area_threshold=50000, max_workers=2):
    """
    画像処理パイプラインをフォルダ単位で並列実行します。
    
    Args:
        max_workers (int): 同時に実行する最大プロセス数。
                           GPUメモリの搭載量に応じて調整してください。
    """
    try:
        # spawnメソッドは、ライブラリ利用側（ノートブック）ではなく、
        # 実際に Process を起動するこの場所で設定するのがより確実です。
        multiprocessing.set_start_method('spawn', force=True)
        print("マルチプロセスの開始メソッドを 'spawn' に設定しました。")
    except RuntimeError:
        # 既に設定されている（2回目以降の呼び出し）場合は何もしない
        pass 

    print(f"並列処理を開始します... 対象: {target_folders} (最大ワーカー数: {max_workers})")
    overall_start_time = time.time()
    
    # --- ▼ ここからが変更点 ▼ ---
    
    # Poolに渡すための引数リストを作成します。
    # (引数が複数あるため、タプルのリストにします)
    tasks = [
        (base_dir, folder, model_path, area_threshold) for folder in target_folders
    ]

    # processes=max_workers で、同時に動くプロセス数を制限します。
    # with構文を使うと、処理完了後に自動でプールを閉じてくれます。
    with multiprocessing.Pool(processes=max_workers) as pool:
        # starmapは、タプルのリスト(tasks)を
        # process_image_subdirectory に展開して渡してくれます。
        # (例: func(arg1, arg2) の代わりに func(*(arg1, arg2)) を実行)
        #
        # この処理は *同期的* で、tasksの全処理が完了するまでここで待機します。
        pool.starmap(process_image_subdirectory, tasks)

    # --- ▲ ここまでが変更点 ▲ ---

    overall_end_time = time.time()
    print(f"\nすべての処理が完了しました。 (総所要時間: {overall_end_time - overall_start_time:.2f}秒)")