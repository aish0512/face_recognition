import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import uuid

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
name = "Aish"
# Reload model 
model = tf.keras.models.load_model('siamesemodel.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    
#resize and rescale the image between 1 and 0 into numpy
def preprocessing(file_path):

    #Read in image from file path
    byte_img = tf.io.read_file(file_path)

    #load in the image
    img = tf.io.decode_jpeg(byte_img)

    #resizing the image
    img = tf.image.resize(img, (100,100))
  
    #rescale between 1 and 0
    img = img/255.0
    return img

def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocessing(os.path.join('application_data', 'input_images', 'input_image.jpg'))
        validation_img = preprocessing(os.path.join('application_data', 'verification_images', image))
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
        
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
    verified = verification > verification_threshold
    
    return results, verified

def face_recog():
     # ASSIGN CAMERA ADDRESS HERE
    camera_id="/dev/video0"
    # Full list of Video Capture APIs (video backends): https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    # For webcams, we use V4L2
    video_capture = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    
    if video_capture.isOpened():
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                # Read the frame
                _, img = cap.read()
                # Detect the faces
                img = img[120:120+250,200:200+250, :]
                faces = face_cascade.detectMultiScale(img, 1.1, 4)
                # Draw the rectangle around each face
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cropped_face = img[y:y+h, x:x+w]
                    if type(cropped_face) is np.ndarray:
                        cv2.imwrite(os.path.join('application_data', 'input_images', 'input_image.jpg'), img)
                        # Run verification
                        results, verified = verify(model, 0.5, 0.5)
                        if verified:
                            cv2.putText(img,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                        else:
                            cv2.putText(img,"No Match", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                    else:
                        cv2.putText(img,"No Face Found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                # Display
                cv2.imshow('img', img)
                # Stop if escape key is pressed
                k = cv2.waitKey(30) & 0xff
                if k==27:
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

    
if __name__ == "__main__":
    face_recog()