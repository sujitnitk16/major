from __future__ import division
import tensorflow as tf, sys
from datetime import datetime
import glob
import os
import csv 
import img_resize
import xlsxwriter
workbook = xlsxwriter.Workbook('Prasent_Students.xlsx')
worksheet = workbook.add_worksheet()
worksheet.set_column('A:A', 20)
worksheet.set_column('B:B', 15)
start_time = datetime.now()
li = []
c=0;
list=[];
i=0
j=1
# Read in the image_data
for filename in glob.glob('person/*.jpg'):
		#new = str(filename).split("/")[6]
		#print new
		#glob("/home/sujit/SK/My_work/data/*/")
		image_data = tf.gfile.FastGFile(filename, 'rb').read()
		print(filename)
		# Loads label file, strips off carriage return
		label_lines = [line.rstrip() for line 
				   in tf.gfile.GFile('tf_files/retrained_labels.txt')]

		# Unpersists graph from file
		with tf.gfile.FastGFile('tf_files/retrained_graph.pb', 'rb') as f:
		    graph_def = tf.GraphDef()
		    graph_def.ParseFromString(f.read())
		    _ = tf.import_graph_def(graph_def, name='')

		with tf.Session() as sess:
		    # Feed the image_data as input to the graph and get first prediction
		    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
		    
		    predictions = sess.run(softmax_tensor, \
			     {'DecodeJpeg/contents:0': image_data})
		    
		    # Sort to show labels of first prediction in order of confidence
		    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
		    a=top_k[0]
		    print('Top Score',predictions[0][a])
		    
		    #print("predicted Person is :",a,label_lines[a])
		    if predictions[0][a]<0.50:
			print("predicted Person is :",'Unknown')
			li.append('Unknown')
     			worksheet.set_row(i, 70)
			worksheet.write('A'+str(j), 'Unknown')
			img=img_resize.resize(filename)
			img='image/'+filename.split('/')[1]
			print img
			worksheet.insert_image('B'+str(j), img)
			print(j)
			i=i+1
			j=j+1
			
		    else:
		    	print("predicted Person is :",label_lines[a])
                        list.append(label_lines[a])
		    	list.append(a);
			worksheet.set_row(i, 70)
			worksheet.write('A'+str(j), label_lines[a])
			img=img_resize.resize(filename)
			img='image/'+filename.split('/')[1]
			print img
			worksheet.insert_image('B'+str(j), img)
			print(j)
			i=i+1
			j=j+1
		    c=c+1;
		    print("Analyze Person identification :")
		    
		    for node_id in top_k:
			human_string = label_lines[node_id]
			score = predictions[0][node_id]
			
			print('%s (score = %.3f)' % (human_string, score))

			
		end_time = datetime.now()
		print('Duration: {}'.format(end_time - start_time))

output = []
for x in list:
    if x not in output:
        output.append(x)


count = 0
for i in output:
    if count % 2 == 0:
        li.append(i)
    count += 1 
print ('Present Students :')
print ('\n'.join(li))
csvfile = 'Present_student.csv'
print("saved in Present_student.csv file")
print("saved in Present_student.xl-sheet file")
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in li:
        writer.writerow([val]) 
