# Agender
A mini project demonstrating how to perform age and gender recognition in real-time using OpenCV's DNN module. The application captures video from the webcam, detects faces in each frame, and predicts the age and gender of the detected faces.

# Prerequisites
Python 3.x
OpenCV
Pre-trained deep learning models for face detection, age prediction, and gender prediction.

# Pre-trained Models
The following pre-trained deep learning models are required for this project:

Face Detection Model: A pre-trained SSD model with MobileNet backbone.
- opencv_face_detector_uint8.pb
- opencv_face_detector.pbtxt
Age Prediction Model: A pre-trained Caffe model.
- age_net.caffemodel
- age_deploy.prototxt
Gender Prediction Model: A pre-trained Caffe model.
- gender_net.caffemodel
- gender_deploy.prototxt

Place these files in the same directory as the Python script.

# Acknowledgements
The pre-trained face detection model is based on the OpenCV deep learning face detector.
The pre-trained age and gender prediction models are based on the WideResNet-16-8 model by Gil Levi and Tal Hassner.
