import numpy as np
import cv2
import os

#list of player names
players = ['bhuvneshwar_kumar','dinesh_karthik','hardik_pandya','jasprit_bumrah','k._l._rahul','kedar_jadhav','kuldeep_yadav','mohammed_shami','ms_dhoni','ravindra_jadeja','rohit_sharma','shikhar_dhawan','vijay_shankar','virat_kohli','yuzvendra_chahal']
count = [37,32,40,21,36,37,45,35,46,49,50,40,31,48,29]
path = "./images/"

# We point OpenCV's CascadeClassifier function to where our 
# classifier (XML file format) is stored
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
imagepaths = [os.path.join(path,f) for f in os.listdir(path)]
len(imagepaths)

cnt = 0
for imagepath in imagepaths:
	cnt1 = 0
	imgpaths = [imagepath+'/'+f for f in os.listdir(imagepath)]
	for imgpath in imgpaths:
		print(imgpath," --> read and processed")
		# Load our image then convert it to grayscale
		image = cv2.imread(imgpath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Our classifier returns the ROI of the detected face as a tuple
		# It stores the top left coordinate and the bottom right coordiantes
		faces = face_classifier.detectMultiScale(gray, 1.3, 5)

		# When no faces detected, face_classifier returns and empty tuple
		if faces is ():
			print("No faces found")
		# We iterate through our faces array and draw a rectangle
		# over each face in faces
		for (x,y,w,h) in faces:
			cv2.rectangle(image, (x,y), (x+w,y+h), (127,0,255), 2)
			cv2.imwrite("./pp_data/Users." + str(cnt) + '.' + str(cnt1) + ".jpg", gray[y:y+h,x:x+w])
			cnt1 = cnt1 + 1
			cv2.waitKey(0)
	cnt = cnt + 1

cv2.destroyAllWindows()
