# Real-Time Object Detector

A computer vision project that detects objects in real time using your laptop webcam, 
built with YOLOv8, OpenCV, and Streamlit.

## What it does
- Opens webcam and detects objects live in browser
- Draws coloured bounding boxes with labels and confidence scores
- Filter which objects to detect from the sidebar
- Adjust minimum confidence threshold
- Live stats panel showing detection counts
- Session summary when you stop



## Tech stack
- Python 3.11
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit


## Setup

Clone the repo
git clone https://github.com/chhavi-29/Object-Detection-Project-.git
cd Object-Detection-Project-

Install dependencies
pip install -r requirements.txt

## Run webcam app (local only)
python -m streamlit run app.py

## Run terminal version
python detect.py

Press Q to quit the terminal version. Session summary prints automatically.

## Detectable objects
person, car, chair, laptop, bottle, dog, cat, bus, phone, bicycle and 70 more.

## Output
- Live detections shown in browser via Streamlit
- All detections saved to detections.txt with timestamps and confidence scores
