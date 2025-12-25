import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.filters import threshold_sauvola

def create_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([120, 40, 255]))
    
    kernel_large = np.ones((20, 20), np.uint8)
    kernel_small = np.ones((10, 10), np.uint8)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=5)
    mask = cv2.erode(mask, kernel_small, iterations=2)
    mask = cv2.dilate(mask, kernel_small, iterations=3)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(mask, [cnt], 0, 255, -1)
    
    return mask

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def detect_corners(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100000:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / w
        
        if 1.3 < aspect_ratio < 4.0:
            valid_contours.append((area, cnt))
    
    if not valid_contours:
        return None
    
    valid_contours.sort(reverse=True)
    largest = valid_contours[0][1]
    
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    corners = order_points(box.astype("float32"))
    return corners

def warp_image(img, corners):
    if corners is None:
        return img
    width = int(max(
        np.linalg.norm(corners[0] - corners[1]),
        np.linalg.norm(corners[2] - corners[3])
    ))
    height = int(max(
        np.linalg.norm(corners[0] - corners[3]),
        np.linalg.norm(corners[1] - corners[2])
    ))
    
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    
    return warped

def crop_image(img):
    mask = create_mask(img)
    corners = detect_corners(mask)
    warped = warp_image(img, corners)

    return warped

def cv2_to_pil(img_cv):
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def setThresholdAdaptive(img, r=31, c=10):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h,s,v = cv2.split(img_hsv)
    
    v = cv2.equalizeHist(v)
    #v = cv2.bilateralFilter(v, 7, 50, 50)
    
    bin_img = cv2.adaptiveThreshold(v, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,
                                    r, c)
    return bin_img


def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def sharpen_image(img):
    gaussian_blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.5, gaussian_blur, -0.5, 0)

# def setThresholdSauvola(img, window=15, k=0.15):
#     # 1. Pojačaj kontrast
#     img = enhance_contrast(img)
    
#     # 2. Oštri sliku
#     img = sharpen_image(img)
    
#     # 3. Upscale (OBAVEZNO za mali font)
#     # PaddleOCR najbolje radi kada je visina karaktera oko 40-50 px
#     height, width = img.shape[:2]
#     img = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # 4. Sauvola sa manjim prozorom za detalje
#     thresh_sauvola = threshold_sauvola(gray, window_size=window, k=k)
#     bin_img = (gray > thresh_sauvola).astype(np.uint8) * 255
    
#     # 5. Invertuj ako je potrebno (PaddleOCR voli crn tekst na beloj pozadini)
#     # Ako ti je pozadina crna a slova bela, dodaj:
#     # bin_img = cv2.bitwise_not(bin_img)

#     return bin_img

def setThresholdSauvola(img, window=25, k=0.2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    # Sauvola 
    thresh_sauvola = threshold_sauvola(gray, window_size=window, k=k)
    bin_img = (gray > thresh_sauvola).astype(np.uint8) * 255

    return bin_img

def warp_img(img_path):

    img = cv2.imread(img_path)
    from docres.preprocess import preprocess_image_for_ocr
    img_small = cv2.resize(img, (500, 500))
    start = time.time()
    result = preprocess_image_for_ocr(img_small)

    return result












    