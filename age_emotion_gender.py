import cv2 as cv
import math
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import sys
import argparse


# specify path to inference models
ageProto = "./deploy_model/deploy_age.prototxt"
ageModel = "./deploy_model/age_net.caffemodel"
genderProto = "./deploy_model/deploy_gender.prototxt"
genderModel = "./deploy_model/gender_net.caffemodel"
emotionModel = './deploy_model/emotion_net.hdf5'
faceModel = './deploy_model/haarcascade_frontalface_alt.xml'

# specify mean image and categorical outputs
MODEL_MEAN_VALUES = (78.426337603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
genderList = ['Male', 'Female']
emotionList = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
font = cv.FONT_HERSHEY_SIMPLEX

# load models
emotionNet = load_model(emotionModel, compile = False)
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)

def age_emotion_gender(image_fp):
	cap = cv.VideoCapture(image_fp)
	padding = 20
	while cv.waitKey(1) < 0:
	    hasFrame, frame = cap.read()
	    if not hasFrame:
	        cv.waitKey()
	        break
	    
	    face_cascade = cv.CascadeClassifier(faceModel)
	    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
	    if(len(faces)>0):
	        print("Found {} faces".format(str(len(faces))))
	        
	    for (x, y, w, h )in faces:
	        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
	        #Get Face 
	        face_img = frame[y:y+h, h:h+w].copy()
	        face_img_gray = gray[y:y+h, h:h+w].copy()
	        blob = cv.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
	    
	        #Predict Gender
	        genderNet.setInput(blob)
	        genderPreds = genderNet.forward()
	        gender = genderList[genderPreds[0].argmax()]
	        print("Gender : " + gender)
	        
	        #Predict Age
	        ageNet.setInput(blob)
	        agePreds = ageNet.forward()
	        age = ageList[agePreds[0].argmax()]
	        print("Age Range: " + age)
	        
	        # Predict Emotion
	        blob = cv.resize(face_img_gray, (64, 64))
	        blob = blob.astype("float") / 255.0
	        blob = img_to_array(blob)
	        blob = np.expand_dims(blob, axis=0)
	        emotionPreds = emotionNet.predict(blob)[0]
	        emotion = emotionList[emotionPreds.argmax()]
	        print("Emotion: " + emotion)
	        
	        overlay_text = "%s %s %s" % (gender, age, emotion)
	        cv.putText(frame, overlay_text, (x, y), font, 0.5, (255, 255, 255), 1, cv.LINE_AA)
	        cv.imshow('frame', frame)
	        cv.imwrite( "./labelled_image/sample.jpg", frame )
	        #0xFF is a hexadecimal constant which is 11111111 in binary.
	        if cv.waitKey(1) & 0xFF == ord('q'): 
	            break

if __name__ =="__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--file", required = True, help="path to test image")
	args = vars(ap.parse_args())
	print(args["file"])
	age_emotion_gender(args["file"])