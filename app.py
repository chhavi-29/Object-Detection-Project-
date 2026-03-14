import streamlit as st
import cv2
from ultralytics import YOLO
from datetime import datetime
import numpy as np

# ── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="Object Detector",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Real-Time Object Detector")
st.caption("Built with YOLOv8 + OpenCV + Streamlit")

# ── SIDEBAR CONTROLS ─────────────────────────────────────
st.sidebar.header("Controls")

all_classes = ["person", "car", "chair", "laptop", "bottle",
               "dog", "cat", "bus", "phone", "bicycle"]

selected_classes = st.sidebar.multiselect(
    "Filter — detect only these objects",
    options=all_classes,
    default=[],
    placeholder="Leave empty to detect everything"
)

min_confidence = st.sidebar.slider(
    "Minimum confidence",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("### How to use")
st.sidebar.markdown("1. Select objects to detect (or leave empty for all)")
st.sidebar.markdown("2. Adjust confidence slider")
st.sidebar.markdown("3. Click **Start Webcam**")
st.sidebar.markdown("4. Click **Stop** to end session")

# ── STATS TRACKING ───────────────────────────────────────
if "detection_counts" not in st.session_state:
    st.session_state.detection_counts = {}
if "session_start" not in st.session_state:
    st.session_state.session_start = None
if "running" not in st.session_state:
    st.session_state.running = False
if "total_logged" not in st.session_state:
    st.session_state.total_logged = 0

# ── LAYOUT ───────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    frame_placeholder = st.empty()

with col2:
    st.markdown("### Live Stats")
    stats_placeholder = st.empty()
    st.markdown("---")
    st.markdown("### Session Summary")
    summary_placeholder = st.empty()

# ── BUTTONS ──────────────────────────────────────────────
btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    start = st.button("▶ Start Webcam", use_container_width=True, type="primary")
with btn_col2:
    stop = st.button("⏹ Stop", use_container_width=True)

# ── MODEL ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ── MAIN LOOP ────────────────────────────────────────────
if start:
    st.session_state.running = True
    st.session_state.detection_counts = {}
    st.session_state.session_start = datetime.now()
    st.session_state.total_logged = 0

if stop:
    st.session_state.running = False

if st.session_state.running:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot open webcam. Check if it is connected and permissions are granted.")
        st.session_state.running = False
    else:
        LOG_FILE = "detections.txt"
        last_log_time = datetime.now()

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot read from webcam.")
                break

            results = model(frame, verbose=False)
            boxes = results[0].boxes
            names = model.names

            detected_labels = []
            detected_confs = []

            for box in boxes:
                label = names[int(box.cls)]
                conf = float(box.conf)

                if conf < min_confidence:
                    continue
                if selected_classes and label not in selected_classes:
                    continue

                detected_labels.append(label)
                detected_confs.append(conf)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
                cv2.putText(frame, f"{label} {conf:.0%}",
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

            # Log every 2 seconds
            current_time = datetime.now()
            if detected_labels and (current_time - last_log_time).seconds >= 2:
                with open(LOG_FILE, "a") as f:
                    ts = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    entries = [f"{l} ({c:.0%})" for l, c in
                               zip(detected_labels, detected_confs)]
                    f.write(f"{ts} — {', '.join(entries)}\n")
                for label in set(detected_labels):
                    st.session_state.detection_counts[label] = \
                        st.session_state.detection_counts.get(label, 0) + 1
                st.session_state.total_logged += 1
                last_log_time = current_time

            # Show frame in browser
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Live stats panel
            if st.session_state.detection_counts:
                stats_text = ""
                for obj, cnt in sorted(st.session_state.detection_counts.items(),
                                       key=lambda x: -x[1]):
                    stats_text += f"**{obj}** — {cnt}\n\n"
                stats_placeholder.markdown(stats_text)
            else:
                stats_placeholder.markdown("No detections yet")

            # Session summary panel
            if st.session_state.session_start:
                elapsed = datetime.now() - st.session_state.session_start
                secs = int(elapsed.total_seconds())
                mins, s = divmod(secs, 60)
                summary_placeholder.markdown(f"""
- Duration: **{mins}m {s}s**
- Log entries: **{st.session_state.total_logged}**
- Unique objects: **{len(st.session_state.detection_counts)}**
""")

        cap.release()