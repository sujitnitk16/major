from PIL import Image
def resize(input_image_path):
	
	output_image_path=input_image_path
	#print output_image_path
	output_image=output_image_path.split('/')[1]
	#print output_image
	output_image_path='image/'+output_image
	#print 'output',output_image_path
	size=(70, 70)
	original_image = Image.open(input_image_path)
	width, height = original_image.size
	resized_image = original_image.resize(size)
	width, height = resized_image.size
	resized_image.save(output_image_path)

