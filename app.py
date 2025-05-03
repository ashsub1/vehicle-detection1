import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Make sure this file is in the same folder

# Detection function
def detect_objects(image):
    results = model(image)
    image_with_boxes = image.copy()

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image_with_boxes

# Streamlit UI
st.set_page_config(page_title="Vehicle Detection Web", layout="wide")
st.title("🚗 Vehicle Detection Web")

option = st.radio("Choose input type:", ("Image", "Video"))

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        output_img = detect_objects(image_np)
        st.image(output_img, caption="Detected Vehicles", use_container_width=True)

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            output_frame = detect_objects(frame)
            stframe.image(output_frame, channels="BGR", use_container_width=True)
