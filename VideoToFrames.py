import cv2
import glob
for filename in glob.glob('Happy_video/*.mp4'):
	print filename
	vidcap = cv2.VideoCapture(filename)
	success,image = vidcap.read()
	count = 0
	success = True

	while success:
		if(success==True):
			success,image = vidcap.read()
			print 'Read a new frame: ', success
			cv2.imwrite("Happy/happy_%d.jpg" % count, image)     
			count += 1
	
