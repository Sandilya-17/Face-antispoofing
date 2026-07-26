import cv2
import numpy as np
import os
from pathlib import Path

def augment_image(img):
    augmented = [img]

    # Small rotation
    for angle in [-10, 10]:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        augmented.append(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT))

    # Brightness/contrast jitter
    for alpha, beta in [(1.1, 10), (0.9, -10)]:
        augmented.append(cv2.convertScaleAbs(img, alpha=alpha, beta=beta))

    # JPEG recompression jitter (simulates splice-boundary artifacts)
    for quality in [50, 75]:
        _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        augmented.append(cv2.imdecode(enc, cv2.IMREAD_COLOR))

    # Slight scale jitter
    h, w = img.shape[:2]
    scale = 1.05
    resized = cv2.resize(img, (int(w*scale), int(h*scale)))
    y0 = (resized.shape[0]-h)//2
    x0 = (resized.shape[1]-w)//2
    augmented.append(resized[y0:y0+h, x0:x0+w])

    return augmented

def run(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        img = cv2.imread(os.path.join(input_dir, fname))
        if img is None:
            continue
        for i, aug in enumerate(augment_image(img)):
            out_name = f"{Path(fname).stem}_aug{i}{Path(fname).suffix}"
            cv2.imwrite(os.path.join(output_dir, out_name), aug)

if __name__ == "__main__":
    run("data/kaggle_final/real", "data/kaggle_final_augmented/real")
    run("data/kaggle_final/fake", "data/kaggle_final_augmented/fake")
