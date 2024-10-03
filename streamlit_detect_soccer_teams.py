import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
import pandas as pd

def calculate_distance(p1, p2, pixel_to_meter_ratio=0.02):
    # Convert pixel distance to meters
    return np.linalg.norm(np.array(p1) - np.array(p2)) * pixel_to_meter_ratio

def detect_soccer_teams(video_path, output_path, model_path):
    # Load the YOLO model from the specified path
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file")
        return

    # Get the video's properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress_bar = st.progress(0)

    def track_passes(frame, prev_positions, curr_positions, pass_threshold=30):
        # Track the number of passes by analyzing player positions across frames
        passes = 0
        for curr_pos in curr_positions:
            for prev_pos in prev_positions:
                dist = calculate_distance(curr_pos, prev_pos)
                if dist <= pass_threshold:
                    passes += 1
        return passes

    def track_distances(prev_positions, curr_positions):
        distances = []
        for curr_pos, prev_pos in zip(curr_positions, prev_positions):
            dist = calculate_distance(curr_pos, prev_pos)
            distances.append(dist)
        return distances

    pass_count = 0
    pass_tracking = []
    prev_positions = []
    distances_covered = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)

        # Extract boxes, confidences, and class IDs for the detected objects
        curr_positions = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # Get bounding box coordinates
            confidence = box.conf[0].cpu().numpy()
            class_id = box.cls[0].cpu().numpy().astype(int)
            if class_id == 0:  # Assuming '0' is the class for soccer team
                curr_positions.append([(x1 + x2) // 2, (y1 + y2) // 2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Soccer Team: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Track passes
        if prev_positions:
            passes = track_passes(frame, prev_positions, curr_positions)
            distances = track_distances(prev_positions, curr_positions)
            pass_count += passes
            pass_tracking.append(pass_count)
            distances_covered.extend(distances)

        prev_positions = curr_positions

        # Write the frame into the output video file
        out.write(frame)

        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

    # Release the video capture and writer
    cap.release()
    out.release()
    st.success("Detection complete. Output video saved.")

    distances_per_frame = pd.DataFrame(distances_covered, columns=["Distance Covered (meters)"], index=range(len(distances_covered)))

    return pass_tracking, distances_per_frame

def main():
    st.title("TSG Öhringen | Soccer Teams Detection, Pass Tracking, and Distance Covered using YOLOv8")
    st.write("Upload a video file to detect soccer teams, track passes, and visualize the distance covered by players.")

    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4"])

    if uploaded_file is not None:
        # Save the uploaded video file to a temporary location
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Perform detection, track passes, and measure distance covered
        model_path = 'yolov8n.pt'  # Model path
        output_path = 'output_video.mp4'
        pass_tracking, distances_covered_df = detect_soccer_teams(tfile.name, output_path, model_path)

        # Stream and display the output video
        st.video(output_path)

        # Display the pass count graph
        if pass_tracking:
            st.write("Pass Tracking Over Time")
            plt.figure(figsize=(10, 5))
            plt.plot(pass_tracking, label='Pass Count')
            plt.xlabel('Frame')
            plt.ylabel('Pass Count')
            plt.legend()
            st.pyplot(plt)

        # Display the distances covered as a table and graph
        if not distances_covered_df.empty:
            st.write("Distances Covered by Players (meters)")
            st.dataframe(distances_covered_df)
            st.write("Distances Covered Over Time (meters)")
            plt.figure(figsize=(10, 5))
            plt.plot(distances_covered_df, label='Distance Covered (meters)')
            plt.xlabel('Frame')
            plt.ylabel('Distance Covered (meters)')
            plt.legend()
            st.pyplot(plt)

if __name__ == "__main__":
    main()
