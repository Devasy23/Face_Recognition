import threading
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
import cv2
import av

# Initialize a list to store the captured images
images = []

# Initialize the count variable
count = 0

# Define a transformer class to process the video frames
def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Stream video from the user's webcam using the webrtc_streamer function
webrtc_streamer(
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    ),
    video_transformer_factory=callback,
    key="unique-stream-key"
)


# Display the captured images
for i, image in enumerate(images):
    st.image(image, width=200)
    st.write(f"Picture {i + 1} of 10")
