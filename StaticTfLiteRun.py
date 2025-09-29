import cv2
from ultralytics import YOLO

# Load your TFLite model
model = YOLO("WasteWiseModel.tflite")
names = model.names

# Load the test image
image_path = "StaticBottles.jpg"  # replace with your image file
frame = cv2.imread(image_path)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Mouse callback to print RGB values
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        b, g, r = frame[y, x]  # OpenCV uses BGR
        print(f"Mouse at [{x}, {y}] - RGB: ({r}, {g}, {b})")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

# Perform detection
results = model(frame)  # this works for TFLite too in Ultralytics API

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

# Show image
cv2.imshow("RGB", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
