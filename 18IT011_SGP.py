import cv2
import numpy as np 

cap = cv2.VideoCapture(0)

_, prev = cap.read()
prev = cv2.flip(prev, 1)  #flips image horizontally
_, new = cap.read()
new = cv2.flip(new, 1) #flips image horizontally


while True:
	diff = cv2.absdiff(prev, new) #gives absolute difference
	diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) #convert image from one color space to another
	diff = cv2.blur(diff, (5,5)) #used to blur the image to smoothen it
	_,thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
	threh = cv2.dilate(thresh, None, 3) #dilation of image, increase the object area
	thresh = cv2.erode(thresh, np.ones((4,4)), 1) #erosion of image, erodes away boundaries of foreground object
	contor,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #contours are joining line
	cv2.circle(prev, (20,200), 5, (0,0,255), -1) #used to draw circle
	for contors in contor:				
		if cv2.contourArea(contors) > 30000:
			(x,y,w,h) = cv2.boundingRect(contors) #rectangle boundreies to moving object
			(x1,y1),rad = cv2.minEnclosingCircle(contors)
			x1 = int(x1)
			y1 = int(y1)
			cv2.line(prev, (20,200), (x1, y1), (255,0,0), 4)
			cv2.putText(prev, "{}".format(int(np.sqrt((x1 - 20)**2 + (y1 - 200)**2))), (100,100),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
			cv2.rectangle(prev, (x,y), (x+w,y+h), (0,255,0), 2)
			cv2.circle(prev, (x1,y1), 5, (0,0,255), -1)
		
					
	
	cv2.imshow("Display", prev)
	
	prev = new
	_, new = cap.read()
	new = cv2.flip(new, 1)

	if cv2.waitKey(1) == 27:
		break

cap.release()
cv2.destroyAllWindows()
