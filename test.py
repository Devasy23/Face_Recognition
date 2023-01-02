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
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from joblib import dump, load
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
    try:
        os.mkdir(name)
    except:
        st.warning("Directory already exists")
        pass
        
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
dhruvil = os.listdir(cwd+"/dhruvil1")
dhruvil.sort()

st.write("Capture complete!")
x= st.button("Start")
if x:
    # download button for test_1.jpg
    # st.write("Download the image below")
    # st.image("test_1.jpg", width=200)
    st.write(files_and_dirs)
    st.write(dhruvil)
    
    
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



def train(name: str, count: int):
    """
    Train a face recognition model using the Eigenface algorithm and an SVM classifier.
    
    Parameters:
    - name: The name of the person whose face will be recognized.
    - count: The number of images to use for training.
    
    Returns:
    - A tuple containing the trained model and the PCA transformer used to preprocess the data.
    """
    # Create a directory to store the training images
    try:
        os.mkdir(name)
    except:
        st.warning("Directory already exists")
        pass
        
    # Store the directory and image count in a JSON file
    data = {"directory": name, "jpg_count": count}
    with open("info.json", "w") as f:
        json.dump(data, f)
    
    # Load the training images and labels
    X_train = []  # images of multiple people
    y_train = np.ones(count)  # labels corresponding to the people in the images (e.g. 0, 1, 2, etc.)
    
    # Train a PCA model
    pca = load('pca.pkl')
    pca.fit(X_train)
    
    # Set up the classification pipeline
    classifier = SVC(kernel='linear', C=1.0)
    
    # Train the classifier on the transformed training data
    X_train_transformed = pca.transform(X_train)
    classifier.fit(X_train_transformed, y_train)
    
    # Save the trained model and PCA transformer
    dump(classifier, 'classifier.pkl')
    dump(pca, 'pca.pkl')
    
    return classifier, pca


def get_next_class_number():
    """
    Get the next available class number based on the number of people stored in the log file.
    """
    # Initialize the class number to 0
    class_number = 0
    
    # Open the log file
    with open("log.json", "r") as f:
        # Load the log file as a dictionary
        log = json.load(f)
    
    # Get the names of all the people stored in the log file
    names = [entry["name"] for entry in log]
    
    # Increment the class number for each person stored in the log file
    for name in names:
        class_number += 1
    
    return class_number

def create_log_file(name: str, file_paths: list, class_: int):
    """
    Create a log file to store information about the captured images.
    Parameters:
    - name: The name of the person whose face is captured in the images.
    - file_paths: A list of file paths for the captured images.
    - class_: The class assigned to the images in the classifier.
    """
    # Create a dictionary to store the log information
    data = {
        "name": name,
        "file_paths": file_paths,
        "class": class_
    }

    # if the log file already exists, append the new data to the existing file
    if os.path.exists("log.json"):
        # Open the log file
        with open("log.json", "r") as f:
            # Load the log file as a dictionary
            log = json.load(f)
        
        # Append the new data to the existing log file
        log.append(data)
        return log
    
    # Write the dictionary to a JSON file
    with open("log.json", "w") as f:
        json.dump(data, f)
        return data
