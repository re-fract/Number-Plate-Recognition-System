import cv2
import numpy as np
import re

def calculate_iou(boxA, boxB):
    #Calc intersection coords
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    #Computing area of both boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if float(boxAArea + boxBArea - interArea) == 0:
        return 0
    #Return IoU to check if the tracker object is same as detected one
    return interArea / float(boxAArea + boxBArea - interArea)

def preprocess_plate(plate_roi):
    #Resizing img
    height, width = plate_roi.shape[:2]
    if height < 50 or width < 100:
        scale = max(100 / width, 50 / height) * 1.5
        plate_roi = cv2.resize(plate_roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    #Apply grayscale
    gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    #Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    #Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    #Thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def ocr_plate(reader, plate_img):
    #Running ocr on the img after preprocessing
    ocr_results = reader.readtext(plate_img)
    for result in ocr_results:
        text = result[1].upper().replace(' ', '')
        conf = result[2]
        if re.match(r'^[A-Z0-9-]{6,10}$', text) and conf > 0.4:
            return text, conf
    return None, 0.0