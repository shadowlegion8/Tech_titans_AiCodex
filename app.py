import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
import pygame  # For playing the siren sound
import base64

# Initialize pygame for sound playback
pygame.mixer.init()

# Load the siren sound file (ensure the path is correct)
siren_sound_path = 'siren.wav'  # Ensure this path is correct
pygame.mixer.music.load(siren_sound_path)

# Load the YOLO model
model = YOLO('yolov8n.pt')

# App title
st.title("Harmful Object Detection App")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Homepage", "Detection"])

# Function to load and encode background image
def load_background_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Homepage background image
try:
    homepage_background_image = load_background_image('homepage_background.jpg')  # Path for homepage background
except FileNotFoundError:
    st.error("Homepage background image not found. Please check the file path.")
    homepage_background_image = None

if page == "Homepage":
    if homepage_background_image:
        # Add a background image using CSS for the homepage
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url('data:image/png;base64,{homepage_background_image}');  /* Use base64 encoded image */
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    st.write("## Welcome to the Harmful Object Detection App!")
    st.write("This application uses YOLO (You Only Look Once) for detecting harmful objects such as knives, guns, and scissors.")
    st.write("You can upload images or videos or use your webcam for real-time detection.")
    st.write("### Instructions:")
    st.write("1. Use the navigation bar on the left to choose between the Homepage and the Detection page.")
    st.write("2. On the Detection page, choose your input type: Image, Video, or Webcam.")
    st.write("3. Follow the prompts to upload your files or start your webcam.")
    st.write("4. The application will detect harmful objects and alert you with a siren sound if any are found.")

else:  # Detection Page
    # Load and encode the background image for the detection page
    try:
        detection_background_image = load_background_image('background.jpg')  # Path for detection page background
    except FileNotFoundError:
        st.error("Detection page background image not found. Please check the file path.")
        detection_background_image = None

    if detection_background_image:
        # Add a background image using CSS for the detection page
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url('data:image/png;base64,{detection_background_image}');  /* Use base64 encoded image */
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    st.write("Upload an image, video, or use your webcam to detect harmful objects (e.g., knives, guns, scissors) using YOLO.")

    # Select input type (image, video, or webcam)
    option = st.radio("Choose input type", ("Image", "Video", "Webcam"))

    # Define harmful classes (ensure these match the YOLO model classes)
    harmful_classes = ['knife', 'gun', 'scissors']

    # Function to play the siren sound
    def play_siren():
        pygame.mixer.music.play()

    # Function to stop the siren sound
    def stop_siren():
        pygame.mixer.music.stop()

    # Process Image Input
    if option == "Image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            # Convert the file to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            # Resize the image to the model input size
            image_resized = cv2.resize(image, (416, 416))

            # Perform object detection
            try:
                results = model(image_resized)
            except Exception as e:
                st.error(f"Error during detection: {e}")
                results = None

            if results is not None:
                # Extract object names from the results
                boxes = results[0].boxes
                detected_objects = [model.names[int(cls)] for cls in boxes.cls]

                # Filter detected harmful objects
                harmful_detected = [obj for obj in detected_objects if obj in harmful_classes]

                if harmful_detected:
                    st.write("Detected harmful objects:", harmful_detected)
                    play_siren()  # Play the siren when harmful objects are detected
                    annotated_image = results[0].plot()  # Draw only harmful objects on the image
                else:
                    st.write("No harmful objects detected.")
                    stop_siren()  # Stop the siren if no harmful objects are detected
                    annotated_image = image  # Show the original image if no harmful objects are detected

                # Display the image with detections
                st.image(annotated_image, caption='Detected Harmful Objects', use_column_width=True)

    # Process Video Input
    elif option == "Video":
        uploaded_video = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
        
        if uploaded_video is not None:
            # Create a temporary file to store the uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_video.read())
                temp_video_path = tfile.name
            
            # Open the video file using OpenCV
            cap = cv2.VideoCapture(temp_video_path)
            stframe = st.empty()
            frame_counter = 0  # Frame counter for skipping frames

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames to speed up detection
                if frame_counter % 2 == 0:  # Process every 2nd frame
                    # Resize the frame to the model input size
                    frame_resized = cv2.resize(frame, (416, 416))

                    # Perform detection on each frame
                    try:
                        results = model(frame_resized)
                    except Exception as e:
                        st.error(f"Error during detection: {e}")
                        break

                    # Extract object names from the results
                    boxes = results[0].boxes
                    detected_objects = [model.names[int(cls)] for cls in boxes.cls]

                    # Filter detected harmful objects
                    harmful_detected = [obj for obj in detected_objects if obj in harmful_classes]

                    if harmful_detected:
                        st.write("Detected harmful objects:", harmful_detected)
                        play_siren()  # Play the siren when harmful objects are detected
                        annotated_frame = results[0].plot()  # Draw only harmful objects on the frame
                    else:
                        stop_siren()  # Stop the siren if no harmful objects are detected
                        annotated_frame = frame  # Show the original frame if no harmful objects are detected

                    # Display video frame by frame
                    stframe.image(annotated_frame, channels="BGR")

                frame_counter += 1  # Increment the frame counter
            
            cap.release()
            
            # Clean up the temporary file
            os.remove(temp_video_path)

    # Process Webcam Input
    elif option == "Webcam":
        st.write("Webcam detection activated. Click 'Start' to begin.")

        # Start button to initiate webcam
        if st.button("Start"):
            cap = cv2.VideoCapture(0)  # Open the webcam
            stframe = st.empty()
            frame_counter = 0  # Frame counter for skipping frames

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames to speed up detection
                if frame_counter % 2 == 0:  # Process every 2nd frame
                    # Resize the frame to the model input size
                    frame_resized = cv2.resize(frame, (416, 416))

                    # Perform detection on each frame
                    try:
                        results = model(frame_resized)
                    except Exception as e:
                        st.error(f"Error during detection: {e}")
                        break

                    # Extract object names from the results
                    boxes = results[0].boxes
                    detected_objects = [model.names[int(cls)] for cls in boxes.cls]

                    # Filter detected harmful objects
                    harmful_detected = [obj for obj in detected_objects if obj in harmful_classes]

                    if harmful_detected:
                        st.write("Detected harmful objects:", harmful_detected)
                        play_siren()  # Play the siren when harmful objects are detected
                        annotated_frame = results[0].plot()  # Draw only harmful objects on the frame
                    else:
                        stop_siren()  # Stop the siren if no harmful objects are detected
                        annotated_frame = frame  # Show the original frame if no harmful objects are detected

                    # Display video frame by frame
                    stframe.image(annotated_frame, channels="BGR")

                frame_counter += 1  # Increment the frame counter

            cap.release()

# Footer
st.markdown("<footer style='text-align: center;'><p>Developed by Tech Titans p></footer>", unsafe_allow_html=True)
