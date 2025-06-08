import cv2
import csv
import re
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import easyocr
from sort.sort import *

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

def draw_plate_box(frame, x1, y1, x2, y2, label):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 4)
    cv2.putText(frame, label, (x1, max(y1-25,0)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 4)

def log_plate(csv_writer, timestamp, plate_text, det_conf, ocr_conf, last_plate_log):
    if plate_text != last_plate_log:
        csv_writer.writerow([timestamp, plate_text, det_conf, ocr_conf])
        return plate_text
    return last_plate_log


def detect_and_ocr_plates(model, reader, frame):
    results = model.predict(frame, device='cuda', conf=0.4)
    detections = []
    current_ocr = {}
    for box in results[0].boxes:
        #Extracting bounding box coords
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        det_conf = float(box.conf[0])
        #Cropping plate from the above coords
        plate_roi = frame[y1:y2, x1:x2]
        if plate_roi.size == 0:
            continue
        #Applying the preprocess function defined above
        thresh = preprocess_plate(plate_roi)
        plate_text, ocr_conf = ocr_plate(reader, thresh)
        #Storing examples
        detections.append([x1, y1, x2, y2, det_conf])
        current_ocr[(x1, y1, x2, y2)] = (plate_text, ocr_conf, det_conf)
    return detections, current_ocr

def update_plate_records(tracks, current_ocr, plate_records):
    active_ids = []
    for track in tracks:
        #Extracting tracker data
        x1, y1, x2, y2, track_id = map(int, track[:5])
        track_id = int(track_id)
        active_ids.append(track_id)
        #Finding best match for it using IoU function above
        best_iou = 0
        best_ocr = (None, 0.0, 0.0)
        for (dx1, dy1, dx2, dy2), ocr_info in current_ocr.items():
            iou = calculate_iou((x1,y1,x2,y2), (dx1,dy1,dx2,dy2))
            if iou > best_iou and iou > 0.3:
                best_iou = iou
                best_ocr = ocr_info
        #Set the best ocr result for that "tracker"/object
        plate_text, ocr_conf, det_conf = best_ocr
        if plate_text:
            existing = plate_records.get(track_id, {'conf':0.0})
            if ocr_conf > existing.get('conf', 0.0):
                plate_records[track_id] = {
                    'text': plate_text,
                    'conf': ocr_conf,
                    'det_conf': det_conf
                }
    #Removing vehicles that leave the frame
    for track_id in list(plate_records.keys()):
        if track_id not in active_ids:
            del plate_records[track_id]
    return plate_records

def draw_and_log_plates(frame, tracks, plate_records, csv_writer, last_plate_log):
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track[:5])
        track_id = int(track_id)
        record = plate_records.get(track_id)
        if record and record['text']:
            #Drawing and logging plates in csv
            label = f"{record['text']} | Det: {record['det_conf']:.2f} | OCR: {record['conf']:.2f}"
            draw_plate_box(frame, x1, y1, x2, y2, label)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            last_plate_log = log_plate(csv_writer, timestamp, record['text'], record['det_conf'], record['conf'], last_plate_log)
    return last_plate_log


def main():
    model = YOLO('runs/detect/train/weights/best.pt')
    reader = easyocr.Reader(['en'], gpu=True)
    tracker = Sort(max_age=15, min_hits=1)
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('examples/sample3.mp4')
    display_size = (1280, 720)

    #Tracks plates
    plate_records = {}
    last_plate_log = None

    with open('examples/detected_plates.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Timestamp', 'Plate_Text', 'Det_Conf', 'OCRxx_Conf'])

        paused = False

        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                #Detect plates and apply ocr
                detections, current_ocr = detect_and_ocr_plates(model, reader, frame)
                #Using np array for tracker
                dets = np.array(detections) if detections else np.empty((0, 5))
                #Update tracker with new detections
                tracks = tracker.update(dets)
                #Updating the best OCR examples per track
                plate_records = update_plate_records(tracks, current_ocr, plate_records)
                last_plate_log = draw_and_log_plates(frame, tracks, plate_records, csv_writer, last_plate_log)
                display_frame = cv2.resize(frame, display_size)
            else:
                display_frame = cv2.resize(frame, display_size)

            cv2.imshow('License Plate Tracking', display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == 32:
                paused = not paused
                if paused:
                    print("Paused. Press 'p' to resume.")
                else:
                    print("Resumed.")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
