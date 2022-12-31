import threading
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
import cv2
import av
import pandas as pd
import pickle
import os
import json
# Initialize a list to store the captured images
images = []

name = "dhruvil1"
def count_jpgfiles(dir):
    count = 0
    for file in os.listdir(dir):
        if file.endswith(".jpg"):
            count += 1
    return count


def store_jpg_count(name, number):
    """
    Stores the number of jpg files in the given directory in a JSON file.
    Parameters:
        directory (str): The directory to search for jpg files.
    """
    # Get the number of jpg files in the given directory
    jpg_count = number
    # make a directory of name = name
    os.mkdir(name)
    # Create a dictionary to store the directory and jpg count
    data = {"directory": name, "jpg_count": jpg_count}

    # Write the dictionary to a JSON file
    with open("info.json", "w") as f:
        json.dump(data, f)

def get_info():
    """
    Get the directory and jpg count from the JSON file.
    """
    # Open the JSON file
    with open("info.json", "r") as f:
        # Load the JSON file
        data = json.load(f)
    # Return the directory and jpg count
    return data["directory"], data["jpg_count"]

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
    name, count = get_info()
    # count+=1
    # number of jpg files in the folder is equal to the count
    cnt= count_jpgfiles((os.getcwd()+"/"+name))
    # img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
    random = np.random.randint(0, 1000)
    file_name_path = f"{name}/{name}_{cnt}.jpg"
    face = face_extractor(img)
    face = cv2.resize(face, (168,192))
    if(cnt > count or face is None):
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    cv2.imwrite(file_name_path, face)
    
    # cv2.imwrite("test.png", img)
    return av.VideoFrame.from_ndarray(face, format="bgr24")

store_jpg_count(name, 10)
# Stream video from the user's webcam using the webrtc_streamer function
webrtc_streamer(key="example", video_frame_callback=callback,rtc_configuration={  # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

     
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
