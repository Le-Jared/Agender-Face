import cv2

# Function to detect faces in a given frame and draw a bounding box around them 
def faceBox(faceNet,frame):
    # Get the height and width of the frame
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    # Create a blob from the input frame
    blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
    # Set the input to the neural network
    faceNet.setInput(blob)
    # Get the output from the neural network
    detection=faceNet.forward()
    # Create an empty list to store the bounding boxes
    bboxs=[]
    # Loop through the detections and store the bounding boxes with a confidence greater than 0.7
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255), 1)
    return frame, bboxs

# Define the paths to the various neural network models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"


# Load the neural network models using OpenCV's DNN module
faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

# Define the mean values used for preprocessing the input to the neural networks
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Define the age and gender classes
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

#Open the video capture device
for i in range(10):
    video = cv2.VideoCapture(i)
    if video.isOpened():
        print(f"Using camera index {i}")
        break

#Set the padding value to be added around the detected face region
padding=20

# Main loop to read frames from the video capture device and process them
while True:
    # Read a frame from the video capture device
    ret,frame=video.read()

    # Call the faceBox function to detect faces in the frame and draw bounding boxes around them
    frame,bboxs=faceBox(faceNet,frame)

    for bbox in bboxs:
        # Extract the face region with padding from the frame
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        
        # Preprocess the face and set it as input to the age and gender networks
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        ageNet.setInput(blob)

        # Perform forward pass to obtain gender and age predictions
        genderPred=genderNet.forward()
        agePred=ageNet.forward()

        # Get the predicted gender and age
        gender=genderList[genderPred[0].argmax()]
        age=ageList[agePred[0].argmax()]

        # Display the predicted gender and age on the frame
        label="{},{}".format(gender,age)
        cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,0,255),-1) 
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
    cv2.imshow("Age-Gender",frame)

    # Break the loop if the 'q' key is pressed
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

# Release the video capture device and close all windows
video.release()
cv2.destroyAllWindows()