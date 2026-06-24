import cv2
import numpy as np
import os
import glob

INPUT_DIR = "/Users/fayyazi/programing/sem4/pcd/fp2/tessss/frames3"
OUTPUT_DIR = "/Users/fayyazi/programing/sem4/pcd/fp2/tessss/frames3-enhanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def enhance(img):
    # GAMMA CORRECTION
    gamma = 1.4335
    table = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255
        for i in range(256)
    ]).astype("uint8")
    result = cv2.LUT(img, table)

    # CLAHE
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0851, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # SATURATION SCALE
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.6021, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result

image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
for path in image_paths:
    filename = os.path.basename(path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(OUTPUT_DIR, f"{name}_enhanced{ext}")
    img = cv2.imread(path)
    if img is None:
        print(f"Gagal load: {path}")
        continue
    result = enhance(img)
    cv2.imwrite(output_path, result)
    print(f"Disimpan: {output_path}")

print("Selesai!")