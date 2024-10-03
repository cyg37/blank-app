import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os

def detect_soccer_teams(video_path, output_path, model_path):
    # Load the YOLO model from the specified path
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get the video's properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)

        # Extract boxes, confidences, and class IDs for the detected objects
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # Get bounding box coordinates
            confidence = box.conf[0].cpu().numpy()
            class_id = box.cls[0].cpu().numpy().astype(int)
            if class_id == 0:  # Assuming '0' is the class for soccer team
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Soccer Team: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame into the output video file
        out.write(frame)

    # Release the video capture and writer
    cap.release()
    out.release()
    print("Detection complete. Output video saved at:", output_path)

def main():
    st.title("Soccer Teams Detection using YOLOv8")
    st.write("Upload a video file to detect soccer teams.")

    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4"])

    if uploaded_file is not None:
        # Save the uploaded video file to a temporary location
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Perform detection
        model_path = 'best.pt'  # Model path
        output_path = 'output_video.mp4'
        detect_soccer_teams(tfile.name, output_path, model_path)

        # Stream and display the output video
        st.video(output_path)

if __name__ == "__main__":
    main()