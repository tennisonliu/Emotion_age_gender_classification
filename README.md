# Emotion_age_gender_classification

# Description
Project to classify age, gender and emotions of human faces in a photo or video stream. 

# Models
The models used to make inferences can be found in /models. The four models and their respective origins:
* Face Detector (openCV Cascade Classifier)
Haar Cascade Classifier.
Note: Only the frontal face classifier is loaded
* Age and Gender Classification Models (Caffe)
Gil Levi and Tal Hassner.Age and Gender Classification Using Convolutional Neural Networks. IEEE Workshop on Analysis and Modeling of Faces and Gestures (AMFG), at the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Boston, 2015.
See: https://talhassner.github.io/home/publication/2015_CVPR
* Emotion Classification (Keras)
Octavio Arriaga, Matias valdenegro-Toro. Real-time Convolutional Neural Networks for Emotion and Gender Classification.
See: https://github.com/oarriaga/face_classification

# Sample Usage
The main file is age_gender_emotion.py. A filepath to the image is supplied on the command line. e.g. python ./age_gender_emotion.py -f ~/test_image/olympic_collage.jpg.
The output is written into /labelled_image.

