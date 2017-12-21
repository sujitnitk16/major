from __future__ import division
import tensorflow as tf, sys
from datetime import datetime
import glob
import os

start_time = datetime.now()

c=0;
list=[];
# Read in the image_data
for filename in glob.glob('person/*.jpg'):
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
		    print("predicted person is :",a,label_lines[a])
		    list.append(a);
		    c=c+1;
		    print("Recognize Person")
		    
		    for node_id in top_k:
			human_string = label_lines[node_id]
			score = predictions[0][node_id]
			
			print('%s (score = %.3f)' % (human_string, score))


		end_time = datetime.now()
		print('Duration: {}'.format(end_time - start_time))
		#print(c)
		if c==16:
			
			#print("---------------each video clip info----------------------")
			print("Final Prediction Person:")
			for i in range(0,len(set(list))):
				a=str(100*(list.count(i)/len(list)))
				print(label_lines[i],a+'%')		
			f=max(list,key=list.count)
			print("Final Prediction Person :",label_lines[f])
			print("---------------------------------------------------------")
			print
			#print("next video clip------------------------------------------")
			c=0;
			list=[]



