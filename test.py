import threading
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings, webrtc_widget
import cv2
import av
import pandas as pd
import pickle

# Initialize a list to store the captured images
images = []

# Initialize the count variable
count = 0

def face_extractor(img):
    # load the face cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # if no face detected
    if faces == ():
        return None
    # crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face

# Define a transformer class to process the video frames
def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    name="test"
    count=1
    img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
    file_name_path = f"{name}_{count}.jpg"
    face = face_extractor(frame)
    face = cv2.resize(face, (168,192))
    cv2.imwrite(file_name_path, face)
    # cv2.imwrite("test.png", img)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Stream video from the user's webcam using the webrtc_streamer function
webrtc_streamer(key="example", video_frame_callback=callback,rtc_configuration={  # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })



st.markdown("## Capture Photo")

# Add a WebRTC widget to the app
webrtc = webrtc_widget(height=400)

# Display the widget
st.write(webrtc)


# Add a button to capture a photo
if st.button("Capture Photo"):
    # Capture the photo and display it
    image = webrtc.capture_frame()
    st.image(image)


# Add a button to download the photo
if st.button("Download Photo"):
    # Convert the image to a png and download it
    image_png = image.to_png()
    st.write(image_png, attachment_type=["png"])


st.write("Capture complete!")
x= st.button("Start")
if x:
    # download button to download the captured images
    for i in range(len(images)):
        st.image(images[i], width=200)
        st.write(f"Picture {i} of {len(images)}")
    st.write(len(images))
    images = np.array(images)
    # st.image(images, width=200)
    st.write(images.shape)
    # st.write(images[0].shape)
    st.write(images)
    st.write("Capture complete!")
