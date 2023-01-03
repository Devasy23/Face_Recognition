import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
from helper import *
from joblib import dump, load



def draw_bounding_box(frame, person):
    # Convert the frame to a NumPy array
    img = frame.to_ndarray(format="bgr24")

    # Load the face cascade
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    # Draw a bounding box around the first face detected
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 60), 2)

        # Display the name of the person above the bounding box
        cv2.putText(img, person, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert the image back to a VideoFrame object and return it
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def wrapper_function(frame):
    pca = load('pca.pkl')
    classf = load('model.pkl')
    # Call the draw_bounding_box function and pass the frame and the name of the person
    # Replace "Person" with the actual name of the person
    frm = frame.to_ndarray(format="bgr24")
    cv2.imwrite("cache.jpg", frm)
    person= predict_image(plt.imread('cache.jpg') , pca, classf, bool=False)
    with open("text.txt", "w") as f:
        f.write(f"{person}")
    person_name = get_name_by_class(person)
    return draw_bounding_box(frame, person_name)



# def draw_bounding_box(frame, person):
#     # Load the face cascade
#     face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#     # Detect faces in the image
#     faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

#     # Draw a bounding box around the first face detected
#     if len(faces) > 0:
#         (x, y, w, h) = faces[0]
#         img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Display the name of the person above the bounding box
#         img = cv2.putText(img, person, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Convert the image back to a VideoFrame object and return it
#     return av.VideoFrame.from_ndarray(img, format="bgr24")

# def testback(frame):
#     '''This test function is called for each frame of the video stream
#     it predicts the face of the person in the frame and displays the name with a bounding box around the face'''
#     img = frame.to_ndarray(format="bgr24")
#     person = predict_image(img)
#     # if person is not None:
#     return draw_bounding_box(frame, person)

    
st.title("Face Recognition System")
mode = st.selectbox("Select a mode", ("Train", "Test"))
 
if mode == "Train":
    # Ask the user for their name
    with st.form(key='my_form'):
        name = st.text_input("Enter your name:")
        num_pics = st.number_input("Enter the number of pictures to capture:", min_value=10, max_value=100, value=50)
        submit_button = st.form_submit_button(label='Submit')
    # Display a warning about proper lighting
    st.warning("Make sure you are in a well-lit area before starting the capture process.")
    
    # Wait for the user to press the "Start" button
    
    if(st.button("Start Capturing ")):
        set_info(name, num_pics)
        try:
            webrtc_streamer(key="example", video_frame_callback=callback)
        except Exception as e: 
            st.error("Please refresh the page and try again", e)
            
        # Get a list of all files and directories in the cwd
        
        if st.button("Train"):
            classn = create_log_file(name, num_pics)
            path = os.getcwd()
            path += "/{}".format(name)

            # Add try-except block before preprocessing for x_train
            try:
                X_train = preprocess_images(path)
            except:
                st.error("Error preprocessing images for x_train. Please refresh the page and try again.")

            st.success("Training the model...")
            class_ = classn

            # Add try-except block before getting test faces
            try:
                x_t = preprocess_images('testfaces')
            except:
                st.error("Error getting test faces. Please refresh the page and try again.")

            y_t = np.full((x_t.shape[0],), 0)
            y_train = np.full((X_train.shape[0],), class_)
            y_train = np.concatenate((y_train, y_t))
            X_train = np.concatenate((X_train, x_t))
            st.write("X_train shape: ", X_train.shape[0])
            
            # Add try-except block before transforming through PCA
            try:
                pca = load('pca.pkl')
                x_pca = pca.transform(X_train)
            except:
                st.error("Error transforming images through PCA. Please refresh the page and try again.")

            classf = SVC(kernel='rbf', class_weight='balanced', C=1e3, gamma=0.001)
            classf.fit(x_pca, y_train)
            dump(classf, 'model.pkl')
            st.success("Model trained successfully")
        

elif mode == "Test":
    # Test the face recognition system using the trained model
    # capture the video stream
    
    # give an option to the user to predict the face of a person from a photo
    option = st.selectbox("Select an option", ("Live-Video", "Upload Image"))
    if option == "Live-Video":
        webrtc_streamer(key="Test Face Recognition", video_frame_callback=wrapper_function)
        # end the video stream
        st.stop()
    elif option == "Upload Image":
        # upload the image
        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if image_file is not None:
            # make predictions
            pca = load('pca.pkl')
            model = load('model.pkl')
            img = predict_image(image_file, pca, model)
            st.write(img)
            pass
        else:
            st.warning("Please upload an image file")
            st.stop()
        
    
    
    
    
    
    
    
    