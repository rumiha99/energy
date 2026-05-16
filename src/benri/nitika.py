import os
from PIL import Image

def binarize_image(image_path, threshold=128):
    image = Image.open(image_path).convert('L')  # 画像をグレースケールに変換
    binary_image = image.point(lambda p: p > threshold and 255)  # 2値化
    return binary_image

def binarize_images_in_folder(folder_path, threshold=128):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            binary_image = binarize_image(image_path, threshold)
            binary_image.save(os.path.join(folder_path, f'{filename}'))

if __name__ == "__main__":
    folder_path = '/home/data/0930/mask'  # 画像が保存されているフォルダのパスを指定
    binarize_images_in_folder(folder_path)