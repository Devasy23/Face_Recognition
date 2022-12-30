import threading
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
import cv2
import av
import pandas as pd
import pickle
import os

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
    # count+=1
    # number of jpg files in the folder is equal to the count
    cnt= count_jpgfiles(os.getcwd())
    # img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
    random = np.random.randint(0, 1000)
    file_name_path = f"{name}_{cnt}.jpg"
    face = face_extractor(img)
    face = cv2.resize(face, (168,192))
    cv2.imwrite(file_name_path, face)
    
    # cv2.imwrite("test.png", img)
    return av.VideoFrame.from_ndarray(face, format="bgr24")

# Stream video from the user's webcam using the webrtc_streamer function
webrtc_streamer(key="example", video_frame_callback=callback,rtc_configuration={  # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

def count_jpgfiles(dir):
    count = 0
    for file in os.listdir(dir):
        if file.endswith(".jpg"):
            count += 1
    return count



cwd = os.getcwd()

# Get a list of all files and directories in the cwd
files_and_dirs = os.listdir(cwd)

st.write("Capture complete!")
x= st.button("Start")
if x:
    # download button for test_1.jpg
    # st.write("Download the image below")
    # st.image("test_1.jpg", width=200)
    st.write(files_and_dirs)
    num = st.number_input("Enter the number of images to be captured", min_value=1, max_value=10, value=1)
    if st.button("show"):
        stringimg = "test_"+str(num)+".jpg"
        st.image(stringimg, width=200)
    
    # st.markdown(get_binary_file_downloader_html("test_1.jpg", "test_1.jpg"), unsafe_allow_html=True)
    
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
