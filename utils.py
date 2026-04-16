import cv2
import numpy as np


def segment_fruit(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = ~((hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 180))

    mask2 = hsv[:, :, 2] > 40

    mask = mask1 & mask2
    mask = mask.astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    if num_labels <= 1:
        return np.ones(mask.shape, dtype=np.uint8) * 255

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    return np.where(labels == largest, 255, 0).astype(np.uint8)


def extract_visual_metrics(path):

    img = cv2.imread(path)
    if img is None:
        raise ValueError("No se pudo leer la imagen")

    img = cv2.resize(img, (224, 224))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = segment_fruit(img)

    h, w = gray.shape
    center_mask = np.zeros_like(gray, dtype=bool)
    center_mask[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)] = True

    fruit_pixels = (mask > 0) & center_mask

    if np.sum(fruit_pixels) < 500:
        fruit_pixels = mask > 0

    dark = gray < 50

    kernel = np.ones((3, 3), np.uint8)
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)

    area_total = np.sum(fruit_pixels)
    stain_area = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > 150:  
            stain_area += area

    stains = stain_area / max(area_total, 1)

    hsv_fruit = hsv[fruit_pixels]

    hue = hsv_fruit[:, 0]
    sat = hsv_fruit[:, 1]
    val = hsv_fruit[:, 2]

    mold_mask = (
        (hue > 35) & (hue < 90) &   
        (sat > 40) &
        (val < 200)
    )

    mold_mask = mold_mask.astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    mold_mask = cv2.morphologyEx(mold_mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mold_mask, 8)

    mold_area = 0
    area_total = len(hsv_fruit)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 50:  # 🔥 clave
            mold_area += area

    mold = mold_area / max(area_total, 1)

    sat_mean = np.mean(hsv[:, :, 1][fruit_pixels]) / 255.0
    wilt = np.clip(1.0 - sat_mean, 0, 1)

    hue_std = np.std(hsv[:, :, 0][fruit_pixels]) / 180.0
    color = np.clip(hue_std, 0, 1)

    return float(stains), float(mold), float(wilt), float(color)