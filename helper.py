# this contains all the helper functions for the main program
 
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

