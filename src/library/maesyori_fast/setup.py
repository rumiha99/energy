# /src/library/maesyori_fast/setup.py

from setuptools import setup, find_packages

setup(
    name="maesyori_fast", # pip list で表示される名前
    version="0.1.0",
    author="oi-hiromu",
    author_email="your.email@example.com",
    description="Shiitake (Maesyori) processing pipeline",
    
    # setup.py と同じ階層にあるパッケージ（__init__.py を含むフォルダ）を探す
    # この場合、'maesyori_fast' フォルダが見つかる
    packages=find_packages(),
    
    # ※ 'package_dir' や 'where' は、この構成では不要です

    python_requires=">=3.10",
    
    # 依存ライブラリ
    install_requires=[
        "torch",
        "ultralytics",
        "opencv-python-headless",
        "numpy",
        "Pillow",
        "tqdm"
    ],
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)