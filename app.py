import streamlit as st
import cv2
import numpy as np
import random
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
from test import *

def train_face_recognition():
    # Ask the user for their name
    name = st.text_input("Enter your name:")
    num_pics = st.number_input("Enter the number of pictures to capture:", min_value=1, max_value=100, value=10)
    # Display a warning about proper lighting
    st.warning("Make sure you are in a well-lit area before starting the capture process.")

    # Wait for the user to press the "Start" button
    if st.button("Start"):
        pass
        



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



def test_face_recognition():
    # Test the face recognition system using the trained model
    pass


st.title("Face Recognition System")
mode = st.selectbox("Select a mode", ("Train", "Test"))
 
if mode == "Train":
    train_face_recognition()
elif mode == "Test":
    test_face_recognition()
