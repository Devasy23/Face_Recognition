# upgraded test code:
import streamlit as st
import cv2
import numpy as np
import random
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
from test import *
from helper import *
 
def testback(frame):
    '''This test function is called for each frame of the video stream
    it predicts the face of the person in the frame and displays the name with a bounding box around the face'''
    img = frame.to_ndarray(format="bgr24")
    person = predict_image(img)
    if person is not None:
        name = person
        cv2.putText(img, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        
    return av.VideoFrame.from_ndarray(img, format="bgr24")

    
st.title("Face Recognition System")
mode = st.selectbox("Select a mode", ("Train", "Test"))
 
if mode == "Train":
    # Ask the user for their name
    name = st.text_input("Enter your name:")
    num_pics = st.number_input("Enter the number of pictures to capture:", min_value=10, max_value=100, value=50)
    # Display a warning about proper lighting
    st.warning("Make sure you are in a well-lit area before starting the capture process.")
    
    set_info(name, num_pics)
    # Wait for the user to press the "Start" button
    if st.button("Start"):
        webrtc_streamer(key="Capture Photos", video_frame_callback=callback)
        # Get a list of all files and directories in the cwd
        path = os.getcwd()
        dir_list = os.listdir(path)
        # Create a list of all the images
        image_list = []
        for file_name in dir_list:
            if file_name.endswith(".jpg"):
                image_list.append(file_name)
        # Create a list of all the cropped faces
        face_list = []
        for image in image_list:
            img = cv2.imread(image)
            face = face_extractor(img)
            if face is not None:
                face = cv2.resize(face, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_list.append(face)
        # Save the cropped faces in a separate directory
        for i in range(len(face_list)):
            file_name_path = "faces/" + name + str(i) + ".jpg"
            cv2.imwrite(file_name_path, face_list[i])
        # Display a message to the user
        st.success("Images captured successfully.")
        
    
elif mode == "Test":
    # Test the face recognition system using the trained model
    # capture the video stream
    
    # give an option to the user to predict the face of a person from a photo
    option = st.selectbox("Select an option", ("Live-Video", "Upload Image"))
    if option == "Live-Video":
        webrtc_streamer(key="Test Face Recognition", video_transformer_factory=testback)
        # end the video stream
        st.stop()
    elif option == "Upload Image":
        # upload the image
        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if image_file is not None:
            # make predictions
            pass
        else:
            st.warning("Please upload an image file")
            st.stop()
        
    
    
    
    
    
    
    
    