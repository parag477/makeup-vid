import streamlit as st
import cv2
import tempfile
import os
from makeup_app import MakeupApplication  # Your MakeupApplication class

# Initialize the makeup application
makeup_app = MakeupApplication()

st.title("Virtual Makeup Application for Uploaded Video")

# Add a file uploader for video files
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Create a temporary file to store the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Open the video file using OpenCV
    video_cap = cv2.VideoCapture(tfile.name)

    # Get video properties
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a placeholder for displaying frames
    video_placeholder = st.empty()

    # Process the video frame by frame
    while video_cap.isOpened():
        success, frame = video_cap.read()
        if not success:
            break

        # Apply the makeup on each frame
        processed_frame = makeup_app.process_frame(frame)

        # Display the processed frame in Streamlit
        video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)

    video_cap.release()

    # Remove temporary file after processing
    os.remove(tfile.name)
