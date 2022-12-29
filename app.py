import streamlit as st
import cv2
import numpy as np
import random


def train_face_recognition():
    # Ask the user for their name
    name = st.text_input("Enter your name:")
    num_pics = st.number_input("Enter the number of pictures to capture:", min_value=1, max_value=100, value=10)
    # Display a warning about proper lighting
    st.warning("Make sure you are in a well-lit area before starting the capture process.")

    # Wait for the user to press the "Start" button
    if st.button("Start"):
        # Initialize the count variable
        count = 0

        # Collect the specified number of samples from the webcam input
        while True:
            # Capture an image using the st.camera_input widget
            img_file_buffer = st.camera_input("Take a pic", key=random.randint(0, 1000000))

            # If the user took a picture, process it
            if img_file_buffer is not None:
                # Convert the image file buffer to a NumPy array
                frame = np.array(img_file_buffer.getvalue(), np.uint8)

                # Extract the face from the frame
                face = face_extractor(frame)

                # If a face was found, save it to a file and display it
                if face is not None:
                    count += 1
                    face = cv2.resize(face, (168,192))

                    # Save the face to a file
                    file_name_path = f"{name}_{count}.jpg"
                    cv2.imwrite(file_name_path, face)

                    # Display the face
                    st.image(face, width=200)

                    # Display the count
                    st.write(f"Picture {count} of {num_pics}")

                # If the required number of pictures has been reached, break the loop
                if count >= num_pics:
                    break
                
        st.write("Capture complete!")



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
