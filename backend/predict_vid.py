from utils import calculate_iou, preprocess_plate, ocr_plate
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from sort.sort import *

class PlateDetector:
    def __init__(self, video_source=0):
        self.model = YOLO('../runs/detect/train/weights/best.pt')
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.tracker = Sort(max_age=15, min_hits=1)
        self.cap = cv2.VideoCapture(video_source)
        self.plate_records = {}
        self.paused = False
        if not self.cap.isOpened():
            print("❌ cv2.VideoCapture FAILED to open video")
        else:
            print("✅ cv2.VideoCapture opened successfully")

    def process_frame(self, frame):
        if not self.paused:
            detections, current_ocr = self._detect_plates(frame)
            tracks = self._update_tracker(detections)
            self._update_records(tracks, current_ocr)
            frame = self._draw_boxes(frame, tracks)
        return frame

    def _detect_plates(self, frame):
        results = self.model.predict(frame, device='cuda', conf=0.4)
        detections = []
        current_ocr = {}

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det_conf = float(box.conf[0])
            plate_roi = frame[y1:y2, x1:x2]

            if plate_roi.size == 0:
                continue

            thresh = preprocess_plate(plate_roi)
            plate_text, ocr_conf = ocr_plate(self.reader, thresh)

            detections.append([x1, y1, x2, y2, det_conf])
            current_ocr[(x1, y1, x2, y2)] = (plate_text, ocr_conf, det_conf)

        return detections, current_ocr

    def _update_tracker(self, detections):
        dets = np.array(detections) if detections else np.empty((0, 5))
        return self.tracker.update(dets)

    def _update_records(self, tracks, current_ocr):
        active_ids = []

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            track_id = int(track_id)
            active_ids.append(track_id)

            best_iou = 0
            best_ocr = (None, 0.0, 0.0)

            for (dx1, dy1, dx2, dy2), ocr_info in current_ocr.items():
                iou = calculate_iou((x1, y1, x2, y2), (dx1, dy1, dx2, dy2))
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_ocr = ocr_info

            plate_text, ocr_conf, det_conf = best_ocr
            if plate_text:
                existing = self.plate_records.get(track_id, {'conf': 0.0})
                if ocr_conf > existing.get('conf', 0.0):
                    self.plate_records[track_id] = {
                        'text': plate_text,
                        'conf': ocr_conf,
                        'det_conf': det_conf
                    }

        for track_id in list(self.plate_records.keys()):
            if track_id not in active_ids:
                del self.plate_records[track_id]

    def _draw_boxes(self, frame, tracks):
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            track_id = int(track_id)
            record = self.plate_records.get(track_id)

            if record and record['text']:
                label = f"{record['text']} | Det: {record['det_conf']:.2f} | OCR: {record['conf']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(frame, label, (x1, max(y1 - 25, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        return frame  # 🟢 make sure this line exists

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow('License Plate Tracking', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 32:  # Spacebar
                self.paused = not self.paused

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = PlateDetector()
    detector.run()
