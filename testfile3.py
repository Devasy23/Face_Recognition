import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av
from helper import *
import os

# filepaths for testfaces
# testfaces = []
# for file in os.listdir("testfaces"):
#     if file.endswith(".jpg"):
#         testfaces.append("testfaces/"+file)

create_log_file('DhruvTest','testfaces')