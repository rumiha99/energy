from PIL import Image
import piexif
import os

def correct_image_rotation(input_image_path, output_image_path):
    image = Image.open(input_image_path)
    exif_dict = piexif.load(image.info['exif'])
    orientation = exif_dict['0th'].get(piexif.ImageIFD.Orientation, 1)
    if orientation == 3:
        image = image.rotate(180, expand=True)
    elif orientation == 6:
        image = image.rotate(270, expand=True)
    elif orientation == 8:
        image = image.rotate(90, expand=True)
    exif_dict['0th'][piexif.ImageIFD.Orientation] = 1
    exif_bytes = piexif.dump(exif_dict)
    image.save(output_image_path, exif=exif_bytes)

def process_images_in_folders(folders):
    for folder in folders:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg','JPEG')):
                    input_image_path = os.path.join(root, file)
                    output_image_path = input_image_path  # ここで出力パスを調整する
                    correct_image_rotation(input_image_path, output_image_path)

folders = [
            '/home/data_org/CD/CD',
            '/home/data_org/A/A',
            '/home/data_org/B/B'
            ]
process_images_in_folders(folders)