import sys
import dlib
import cv2
from skimage import io
from datetime import datetime
import label_images_mod
j=0
def draw_text(img, text, x,y,t,b):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

if len(sys.argv) < 3:
    print(
        "Call this program like this:\n"
        "   ./cnn_face_detector.py mmod_human_face_detector.dat ../examples/faces/*.jpg\n"
        "You can get the mmod_human_face_detector.dat file from:\n"
        "    http://dlib.net/files/mmod_human_face_detector.dat.bz2")
    exit()
start_time = datetime.now()
cnn_face_detector = dlib.cnn_face_detection_model_v1(sys.argv[1])
win = dlib.image_window()

for f in sys.argv[2:]:
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
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
            i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
	#rect=[d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()]
        #print ("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
        cv2.rectangle(img, (d.rect.left(), d.rect.top()), (d.rect.right(), d.rect.bottom()), (0,0,255), 2)
        crop = img[d.rect.top():d.rect.bottom(), d.rect.left():d.rect.right()]
	face_file_name = "person/r_"+str(j)+".jpg"
	cv2.imwrite(face_file_name, crop)
	j=j+1
	draw_text(img,label_images_mod.recognize(face_file_name),d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom())
	cv2.imwrite(face_file_name, crop)
    rects = dlib.rectangles()
    rects.extend([d.rect for d in dets])
    cv2.imshow("img",img)
    #win.clear_overlay()
    #win.set_image(img)
    #win.add_overlay(rects)
    
    cv2.imwrite('out.jpg', img)
    
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
dlib.hit_enter_to_continue()
