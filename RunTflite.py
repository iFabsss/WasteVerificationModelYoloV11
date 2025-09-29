import cv2
from ultralytics import YOLO

# Load your TFLite model
model = YOLO("WasteWiseModel.tflite")
names = model.names

# Start webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

# Optional: mouse callback to print RGB values
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        b, g, r = frame[y, x]  # OpenCV uses BGR
        print(f"Mouse at [{x}, {y}] - RGB: ({r}, {g}, {b})")

cv2.namedWindow("YOLO TFLite Webcam")
cv2.setMouseCallback("YOLO TFLite Webcam", RGB)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)  # works for TFLite in Ultralytics API

    # Draw detections
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        scores = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{names[cls]} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show webcam feed
    cv2.imshow("YOLO TFLite Webcam", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
