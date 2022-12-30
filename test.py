import threading
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings

# Initialize a list to store the captured images
images = []

# Initialize the count variable
count = 0

# Define a transformer class to process the video frames
class FrameCapturer(VideoTransformerBase):
    def __init__(self):
        self.count = 0

    def transform(self, frame):
        # Convert the frame to a NumPy array
        image = frame.to_ndarray(format="bgr24")

        # Add the image to the list of images
        images.append(image)

        # Increment the count
        self.count += 1

        # If we have collected 10 images, stop capturing
        if self.count == 10:
            return None

        return image

# Stream video from the user's webcam using the webrtc_streamer function
webrtc_streamer(
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    ),
    video_transformer_factory=FrameCapturer,
)

# Display the captured images
for i, image in enumerate(images):
    st.image(image, width=200)
    st.write(f"Picture {i + 1} of 10")
