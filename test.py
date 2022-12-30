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

def face_extractor(frame):
    # load the face cascade
    img = frame.to_ndarray(format="bgr24")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # if no face detected
    if faces == ():
        return None
    # crop all faces found
    print(type(faces))
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return av.VideoFrame.from_ndarray(cropped_face, format="bgr24")

# Define a transformer class to process the video frames
def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Stream video from the user's webcam using the webrtc_streamer function
webrtc_streamer(key="example", video_frame_callback=callback,rtc_configuration={  # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })



 
