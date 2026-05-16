import os
import cv2
import numpy as np
from ultralytics import YOLO

class Yolo_crop:
    def __init__(self, model_path, save_dir):
        self.model = YOLO(model_path)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def crop_object(self, image_path):
        image = cv2.imread(image_path)
        results = self.model(image_path)[0]  # 最初の結果のみ使用

        for i, box in enumerate(results.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            cropped = image[y1:y2, x1:x2]
            save_path = os.path.join(self.save_dir, f"crop_{i}.jpg")
            cv2.imwrite(save_path, cropped)
            print(f"✔ Cropped image saved to: {save_path}")
            yield save_path, cropped

class Yolo_mask:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def generate_mask(self, crop_path, save_dir, image_array=None):
        os.makedirs(save_dir, exist_ok=True)

        if image_array is None:
            image_array = cv2.imread(crop_path)

        results = self.model(image_array)[0]
        masks = results.masks.data.cpu().numpy()

        for i, mask in enumerate(masks):
            bin_mask = (mask * 255).astype('uint8')
            save_path = os.path.join(save_dir, f"mask_{os.path.basename(crop_path)}")
            cv2.imwrite(save_path, bin_mask)
            print(f"✔ Mask image saved to: {save_path}")

# クラスのインスタンス化時に save_dir を渡さない
cropper = Yolo_crop(
    model_path="/home/YOLO/hukusuu_train/datasets/train7/weights/best.pt",
    save_dir="/home/test_hozon"
)

segmenter = Yolo_mask(
    model_path="/home/YOLO/-327_seg/datasets/train2/weights/best.pt"
)

image_path = "/home/data/maesyori_img/collage_1.jpg"

for crop_path, cropped_img in cropper.crop_object(image_path):
    segmenter.generate_mask(crop_path, save_dir="/home/test_hozon", image_array=cropped_img)


class MaskCreator:
    def __init__(self, lower_hsv=None, upper_hsv=None):
        self.lower_hsv = lower_hsv if lower_hsv is not None else np.array([20, 100, 100])
        self.upper_hsv = upper_hsv if upper_hsv is not None else np.array([255, 255, 255])

    def create_mask(self, image_bgr):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        return cv2.bitwise_not(mask)


class BB_crop:
    def __init__(self, padding=30, save_dir="./cropped_images"):
        self.padding = padding
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(self, image_bgr, mask, filename="cropped.jpg"):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found")

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        height, width = image_bgr.shape[:2]
        x1 = max(x - self.padding, 0)
        y1 = max(y - self.padding, 0)
        x2 = min(x + w + self.padding, width)
        y2 = min(y + h + self.padding, height)

        cropped = image_bgr[y1:y2, x1:x2]
        save_path = os.path.join(self.save_dir, filename)
        cv2.imwrite(save_path, cropped)
        return save_path


class BB_mask:
    def __init__(self, padding=30, save_dir="./cropped_masks"):
        self.padding = padding
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(self, mask, filename="cropped_mask.jpg"):
        # maskがカラー画像の場合、グレースケールに変換
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # マスクがすでにバイナリ画像でない場合、バイナリ化
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # コンターの取得
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found")

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        height, width = mask.shape[:2]
        x1 = max(x - self.padding, 0)
        y1 = max(y - self.padding, 0)
        x2 = min(x + w + self.padding, width)
        y2 = min(y + h + self.padding, height)

        cropped_mask = mask[y1:y2, x1:x2]
        save_path = os.path.join(self.save_dir, filename)
        cv2.imwrite(save_path, cropped_mask)
        return save_path

class BB2BBmask:
    def __init__(self, lower_hsv=None, upper_hsv=None, padding=30):
        # マスクを作成するためのHSV範囲を指定
        self.lower_hsv = lower_hsv if lower_hsv is not None else np.array([20, 100, 100])
        self.upper_hsv = upper_hsv if upper_hsv is not None else np.array([255, 255, 255])
        self.padding = padding

    def create_mask(self, image_bgr):
        # BGRからHSVに変換
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        # 指定したHSV範囲にマッチする部分のマスクを作成
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        # 反転マスク（対象物を白、背景を黒にする）
        mask_inv = cv2.bitwise_not(mask)
        return mask_inv

    def crop_image(self, image_bgr, mask_inv):
        height, width = image_bgr.shape[:2]

        # 輪郭を検出
        contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found")

        # 最大の輪郭を取得
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # パディングを追加（画像サイズを超えないように調整）
        x_pad = max(x - self.padding, 0)
        y_pad = max(y - self.padding, 0)
        x2_pad = min(x + w + self.padding, width)
        y2_pad = min(y + h + self.padding, height)

        # 切り取る
        cropped_image = image_bgr[y_pad:y2_pad, x_pad:x2_pad]
        cropped_mask = mask_inv[y_pad:y2_pad, x_pad:x2_pad]
        return cropped_image, cropped_mask

    def save_mask(self, cropped_mask, mask_output_path):
        # マスク画像のみを保存
        cv2.imwrite(mask_output_path, cropped_mask)

image_path = '/home/data/0203_energee_after/org/B/IMG_1703.JPEG'
filename = os.path.basename(image_path)

image = cv2.imread(image_path)

# クラスインスタンス生成
mask_creator = MaskCreator()
mask = mask_creator.create_mask(image)

# 保存クラス（画像、マスク）
image_saver = BB_crop(padding=30, save_dir="/home/test_hozon")
mask_saver = BB_mask(padding=30, save_dir="/home/test_hozon")

# 保存
image_saver.save(image, mask, filename="cropped_" + filename)
mask_saver.save(mask, filename="mask_" + filename)



