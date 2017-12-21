import sys
import dlib
from skimage import io
import glob
import cv2
from datetime import datetime
j=0
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
win = dlib.image_window()
start_time = datetime.now()
for f in glob.glob('faces/*.jpg'):
    print("Processing file: {}".format(f))
    
    img = io.imread(f)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    
    dets = cnn_face_detector(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    '''
    This detector returns a mmod_rectangles object. This object contains a list of mmod_rectangle objects.
    These objects can be accessed by simply iterating over the mmod_rectangles object
    The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score.
    
    It is also possible to pass a list of images to the detector.
        - like this: dets = cnn_face_detector([image list], upsample_num, batch_size = 128)
    In this case it will return a mmod_rectangless object.
    This object behaves just like a list of lists and can be iterated over.
    '''
    print("Number of faces detected: {}".format(len(dets)))
    '''for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
            i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))'''
    for i, d in enumerate(dets):
		#print("Detection {}, score: {}, face_type:{}".format(d, scores[i], idx[i]))
		crop = img[d.rect.top():d.rect.bottom(), d.rect.left():d.rect.right()]
		face_file_name = "person/r_"+str(j)+".jpg"
		j=j+1
		#print face_file_name
		#img = cv2.cvtColor(face_file_name, cv2.COLOR_BGR2RGB)
		cv2.imwrite(face_file_name, crop)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

'''
    rects = dlib.rectangles()
    rects.extend([d.rect for d in dets])

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(rects)
'''
