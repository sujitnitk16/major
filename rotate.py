import PIL, os
from PIL import Image
import glob
#os.chdir('/home/itadmin/Internship/atten/sujit') # change to directory where image is located
for filename in glob.glob('Happy_frames/*.jpg'):
	picture= Image.open(filename)
	print filename
	picture.rotate(270).save(filename)

