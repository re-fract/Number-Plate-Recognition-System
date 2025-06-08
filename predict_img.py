from ultralytics import YOLO
import cv2
import easyocr

model = YOLO('runs/detect/train/weights/best.pt')

image_path = 'examples/image.jpg'
image = cv2.imread(image_path)

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

results = model(image)
result = results[0]
reader = easyocr.Reader(['en'])

for box in result.boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    plate_roi = image[y1:y2, x1:x2]

    if plate_roi.size == 0:
        continue

    preprocessed = preprocess_plate(plate_roi)
    ocr_results = reader.readtext(preprocessed)

    ocr_text = ocr_results[0][1]

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, ocr_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()