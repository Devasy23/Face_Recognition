# this contains all the helper functions for the main program
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np
import pickle
import json
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from joblib import dump, load
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
import streamlit as st
import av
import threading
import time
import random

def set_info(name, number):
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

def predict_face(face):
    face = np.mean(face,axis=2).T.flatten()
    face = face.reshape(1,-1)
    face = pca.transform(face)
    prediction = classf.predict(face)
    return int(prediction[0])

def predict_image(image):
    # load the face cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # convert to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # if no face detected
    
    if faces is None:
        return None
    # crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = image[y:y+h, x:x+w]
    face = cv2.resize(cropped_face, (168,192))

    return predict_face(face)

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

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    name, count = get_info()
    # count+=1
    # number of jpg files in the folder is equal to the count
    cnt= count_jpgfiles((os.getcwd()+"/"+name))
    file_name_path = f"{name}/{name}_{cnt}.jpg"
    face = face_extractor(img)
    face = cv2.resize(face, (168,192))
    if(cnt > count or face is None):
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    try:
        cv2.imwrite(file_name_path, face)
    except:
        st.warning("Error saving image")
        pass
    # cv2.imwrite("test.png", img)
    return av.VideoFrame.from_ndarray(face, format="bgr24")

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
    # for y train we need to create a list of labels
    y_train = []  # assign class labels based on the directory name
    # get the directory and jpg count from the JSON file
    
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

def preprocess_image(pathofimg, target_size):
    """
    Preprocess an image to be used as input to the face recognition model.
    Parameters:
    - image: The image to preprocess.
    - target_size: The size to which the image will be resized.
    Returns:
    - The preprocessed image.
    """
    tester_face = (plt.imread(pathofimg))
    tester_face = np.mean(tester_face,axis=2).T.flatten()
    tester_face = tester_face.reshape(1,-1)
    return tester_face

def preprocess_images(path):
    """
    Preprocess an image to be used as input to the face recognition model.
    Parameters:
    - directory: The directory containing the images to preprocess.
    Returns:
    - The preprocessed image.
    """
    custom_images = []
    input_faces = os.listdir(path)
    for face in input_faces:
        testFace = (plt.imread('testfaces/'+face))
        # preprocess faces to match the size of the training data
        testFace = np.mean(testFace,axis=2).T.flatten()
        custom_images.append(testFace)
        # print(testFace.shape)


    # concatenate custom faces with original faces
    custom_images = np.array(custom_images)
    custom_images = custom_images.T
    return custom_images