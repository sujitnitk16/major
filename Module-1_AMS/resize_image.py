from PIL import Image
import glob
def resize(input_image_path):
	for f in glob.glob(input_image_path):
		output_image_path=f
		#print output_image_path
		#output_image=output_image_path.split('/')[1]
		#print output_image
	
		#print 'output',output_image_path
		size=(1024, 1024)
		original_image = Image.open(f)
		width, height = original_image.size
	
		resized_image = original_image.resize(size)
		width, height = resized_image.size
		#print(output_image_path)
		resized_image.save(output_image_path)
resize("faces/*.jpg")

