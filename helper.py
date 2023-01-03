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

def get_name(number):
    # load log.json
    with open("log.json", "r") as f:
        data = json.load(f)
    # get the name from the log.json
    name = data[number]["name"]
    return name
    

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

def predict_face(face, pca, classf):
    face = np.mean(face,axis=2).T.flatten()
    face = face.reshape(1,-1)
    face = pca.transform(face)
    prediction = classf.predict(face)
    return int(prediction[0])

def predict_image(image, pca, classf, bool=True):
    if(bool):
        image = plt.imread(image)
    # load the face cascade

    if face_extractor(image) is None:
        return None
    # crop all faces found
    face = cv2.resize(face_extractor(image), (168,192))
    return predict_face(face, pca, classf)

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

def get_name_by_class(class_number: int):
    """
    Get the name of the person with the given class number from the log file.
    
    Parameters:
    - class_number: The class number to search for.
    
    Returns:
    - The name of the person with the given class number, or None if no such person is found.
    """
    # Open the log file
    with open("log.json", "r") as f:
        # Load the log file as a dictionary
        log = json.load(f)
    
    # Search for an entry with the given class number
    for entry in log["User_details"]:
        if entry["Class"] == class_number:
            return entry["Name"]
    
    # If no entry was found, return None
    return None

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
    try:
        with open("log.json", "r") as f:
            # Load the log file as a dictionary
            log = json.load(f)
        # Get the list of user details from the log file
        users = log["User_details"]
        # Get the number of users stored in the log file
        num_users = len(users)
        # Set the class number to the number of users stored in the log file
        class_number = num_users
        
    except:
        # If the log file doesn't exist, create an empty dictionary
        log = {"User_details": []}

    return class_number

def create_log_file(name: str, count: int, filename="log.json"):
    ''' Creates a log file with the name of the person and the number of images used for training.
    Parameters:
        name (str): The name of the person whose face will be recognized.
        count (int): The number of images to use for training.
        filename (str): The name of the log file.
    Returns:
        class_ (int): The class number assigned to the person.
         
    '''
    class_ = get_next_class_number()
    y = {"Name": name,
	"Count" : count,
    "Class" : class_
	}
    with open(filename,'r+') as file:
        file_data = json.load(file)
        file_data["User_details"].append(y)
        file.seek(0)
        json.dump(file_data, file, indent = 4)
    return class_

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
        testFace = (plt.imread(path+'/'+face))
        # preprocess faces to match the size of the training data
        testFace = np.mean(testFace,axis=2).T.flatten()
        custom_images.append(testFace)
        # print(testFace.shape)


    # concatenate custom faces with original faces
    custom_images = np.array(custom_images)
    custom_images = custom_images.T
    if(custom_images.shape[0] == 32256):
        custom_images = custom_images.T
    return custom_images

