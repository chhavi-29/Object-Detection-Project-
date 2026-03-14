import cv2
from ultralytics import YOLO
from datetime import datetime

# ── CONFIG ───────────────────────────────────────────────
print("\n--- Object Detection Filter ---")
print("Available classes: person, car, chair, laptop, bottle, dog, cat, bus, phone, bicycle")
print("Type the objects you want to detect separated by commas.")
print("Or just press Enter to detect everything.\n")

user_input = input("What do you want to detect? ").strip()

if user_input == "":
    FILTER_CLASSES = []
    print("No filter set — detecting all objects.\n")
else:
    FILTER_CLASSES = [c.strip().lower() for c in user_input.split(",")]
    print(f"Filter set — detecting only: {FILTER_CLASSES}\n")

LOG_FILE = "detections.txt"
MIN_CONFIDENCE = 0.5  # ignore anything below 50% confidence

# ── MODEL LOAD ───────────────────────────────────────────
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam. Check if it is connected.")
    exit()

print("Webcam running. Press Q to quit.")

# ── LOGGING + STATS (Person 3) ───────────────────────────
detection_counts = {}
session_start = datetime.now()

def log_detections(labels, confidences):
    global detection_counts
    with open(LOG_FILE, "a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entries = [f"{l} ({c:.0%})" for l, c in zip(labels, confidences)]
        f.write(f"{ts} — {', '.join(entries)}\n")
    for label in set(labels):
        detection_counts[label] = detection_counts.get(label, 0) + 1

def print_summary():
    duration = datetime.now() - session_start
    seconds = int(duration.total_seconds())
    mins, secs = divmod(seconds, 60)
    total = sum(detection_counts.values())

    print("\n---- Session Summary ----")
    print(f"Duration        : {mins} min {secs} sec")
    print(f"Total detections: {total}")

    if detection_counts:
        top = max(detection_counts, key=detection_counts.get)
        print(f"Most detected   : {top} ({detection_counts[top]} times)")
        print(f"Unique objects  : {', '.join(detection_counts.keys())}")
        print("\nFull breakdown:")
        for obj, count in sorted(detection_counts.items(), key=lambda x: -x[1]):
            print(f"  {obj}: {count}")
    print("-------------------------\n")

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
    detected_confs = []

    for box in boxes:
        label = names[int(box.cls)]
        conf = float(box.conf)

        # Skip low confidence detections
        if conf < MIN_CONFIDENCE:
            continue

        # Skip if not in filter
        if FILTER_CLASSES and label not in FILTER_CLASSES:
            continue

        detected_labels.append(label)
        detected_confs.append(conf)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
        cv2.putText(frame, f"{label} {conf:.0%}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

    # Show per-object cumulative counts on screen
    y_pos = 40
    for obj, cnt in detection_counts.items():
        cv2.putText(frame, f"{obj}: {cnt}",
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        y_pos += 30

    # Log every 2 seconds only
    current_time = datetime.now()
    if detected_labels:
        if not hasattr(log_detections, "last_log") or \
           (current_time - log_detections.last_log).seconds >= 2:
            log_detections(detected_labels, detected_confs)
            log_detections.last_log = current_time

    cv2.imshow("Object Detector — cv-project", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print_summary()
print("Check detections.txt for full logs.")