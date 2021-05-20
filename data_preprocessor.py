import numpy as np
import cv2

#list of player names
players = ['bhuvneshwar_kumar','dinesh_karthik','hardik_pandya','jasprit_bumrah','k._l._rahul','kedar_jadhav','kuldeep_yadav','mohammed_shami','ms_dhoni','ravindra_jadeja','rohit_sharma','shikhar_dhawan','vijay_shankar','virat_kohli','yuzvendra_chahal']
count = [37,32,40,21,36,37,45,35,46,49,50,40,31,48,29]

# We point OpenCV's CascadeClassifier function to where our 
# classifier (XML file format) is stored
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for i in range(15):
	cnt = 0
	for j in range(count[i]):
		# Load our image then convert it to grayscale
		image = cv2.imread('./images/'+players[i]+'/'+str(j+1)+'.jpg')
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
			cv2.imwrite("./pp_data/Users." + str(i) + '.' + str(cnt) + ".jpg", gray[y:y+h,x:x+w])
			cnt = cnt + 1
			cv2.waitKey(0)

cv2.destroyAllWindows()