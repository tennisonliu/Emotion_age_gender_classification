import cv2 as cv
import math
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import sys
import argparse
import dlib

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

faceTrackers = {}
facePreds = {}

def detect_aeg(face_img):
	face_img_gray = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
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

	return gender, age, emotion

def fid_compare(x_bar, y_bar, x, y, w, h):
	# check if centerpoint is in tracker and vice versa
	matchedFid = None
	for fid in faceTrackers.keys():
		tracked_position = faceTrackers[fid].get_position()

		t_x = int(tracked_position.left())
		t_y = int(tracked_position.top())
		t_w = int(tracked_position.width())
		t_h = int(tracked_position.height())
		t_x_bar = t_x + 0.5 * t_w
		t_y_bar = t_y + 0.5 * t_h

		if ( ( t_x <= x_bar   <= (t_x + t_w)) and 
			 ( t_y <= y_bar   <= (t_y + t_h)) and 
			 ( x   <= t_x_bar <= (x   + w  )) and 
			 ( y   <= t_y_bar <= (y   + h  ))):
			matchedFid = fid

	return matchedFid

def main(image_fp):
	cap = cv.VideoCapture(0)
	frameCounter = 0
	currentFaceID = 0
	while cv.waitKey(1) < 0:
		hasFrame, frame = cap.read()
		if not hasFrame:
			cv.waitKey()
			break
		frameCounter += 1
		fidsToDelete = []
		for fid in faceTrackers.keys():
			trackingQuality = faceTrackers[fid].update(frame)

			if trackingQuality < 7:
				fidsToDelete.append(fid)

		for fid in fidsToDelete:
			print('Removing fid {} from list of trackers'.format(fid))
			faceTrackers.pop(fid, None)

		if (frameCounter%10) == 0:
			face_cascade = cv.CascadeClassifier(faceModel)
			gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(gray, 1.1, 5)
			if(len(faces)>0):
				print("Found {} faces".format(str(len(faces))))
				
			for (x, y, w, h )in faces:
				# cv.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
				x_bar = x + 0.5*w
				y_bar = y + 0.5*h
				# Loop over all the trackers and check if the centerpoint
				# of face is within the box of a tracker
				matchedFid = fid_compare(x_bar, y_bar, x, y, w, h)
				face_img = frame[y:y+h, h:h+w].copy()
				if matchedFid is None:
					print('Creating new tracker ' + str(currentFaceID))

					# create and store tracker
					tracker = dlib.correlation_tracker()
					tracker.start_track(frame, dlib.rectangle(x-10,
						y - 20, x+w+10, y+h+20))
					faceTrackers[currentFaceID] = tracker
					(gender, age, emotion) = detect_aeg(face_img)
					facePreds[currentFaceID] = [gender, age, emotion]
					currentFaceID += 1
				else:
					(gender, age, emotion) = detect_aeg(face_img)
					facePreds[matchedFid] = [gender, age, emotion]

		for fid in faceTrackers.keys():
			tracked_position = faceTrackers[fid].get_position()
			t_x = int(tracked_position.left())
			t_y = int(tracked_position.top())
			t_w = int(tracked_position.width())
			t_h = int(tracked_position.height())
			cv.rectangle(frame, (t_x, t_y), (t_x+t_w, t_y+t_h), (255, 255, 0), 2)

			(gender, age, emotion) = facePreds[fid]
			overlay_text = "Face ID: %s, %s, %s, %s" % (fid, gender, age, emotion)
			cv.putText(frame, overlay_text, (t_x, t_y), font, 0.5, (255, 255, 255), 1, cv.LINE_AA)
			cv.imshow('frame', frame)
			cv.imwrite( "./labelled_image/sample.jpg", frame )

		if cv.waitKey(1) & 0xFF == ord('q'): 
				break

if __name__ =="__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--file", required = False, help="path to test image")
	args = vars(ap.parse_args())
	print(args["file"])
	main(args["file"])