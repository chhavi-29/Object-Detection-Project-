import cv2
from ultralytics import YOLO
from datetime import datetime

# ── CONFIG ───────────────────────────────────────────────
# Person 2 will fill FILTER_CLASSES later e.g. ["person","car"]
FILTER_CLASSES = []
LOG_FILE = "detections.txt"

# ── MODEL LOAD ───────────────────────────────────────────
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam. Check if it is connected.")
    exit()

print("Webcam running. Press Q to quit.")

# ── LOGGING FUNCTION (Person 3 will expand this) ─────────
def log_detections(labels):
    with open(LOG_FILE, "a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts} — {', '.join(labels)}\n")

# ── MAIN DETECTION LOOP ──────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read from webcam.")
        break

    results = model(frame, verbose=False)
    boxes = results[0].boxes
    names = model.names

    detected_labels = []

    for box in boxes:
        label = names[int(box.cls)]

        # Filter logic — Person 2 will activate this
        if FILTER_CLASSES and label not in FILTER_CLASSES:
            continue

        detected_labels.append(label)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
        cv2.putText(frame, f"{label} {conf:.0%}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

    # Show count on screen
    cv2.putText(frame, f"Objects detected: {len(detected_labels)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Log if anything detected
    if detected_labels:
        log_detections(detected_labels)

    cv2.imshow("Object Detector — cv-project", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Session ended. Check detections.txt for logs.")