import sys
import os
import numpy as np
import math
from operator import itemgetter, attrgetter, methodcaller
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import DropoutWrapper
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import time



def getBatch(X, Y, i, batch_size):
    start_id = i*batch_size
    end_id = min( (i+1) * batch_size, X.shape[0])
    batch_x = X[start_id:end_id]
    batch_y = Y[start_id:end_id]
    return batch_x, batch_y


def checkTest(ts_data, batchsz, label_test):
	tot_pred = []
#	gt_test = []
	iterations = ts_data.shape[0] / batchsz

	if ts_data.shape[0] % batchsz != 0:
	    iterations+=1
	
	begin = time.time()
	for ibatch in range(iterations):
		batch_rnn_x, batch_y = getBatch(ts_data, label_test, ibatch, batchsz)

		pred_temp = sess.run(testPrediction,feed_dict={x_rnn:batch_rnn_x})
		
		del batch_rnn_x
		del batch_y
		
		for el in pred_temp:
			tot_pred.append( el )
			
	end = time.time()
	elapsed = end - start
	print "ELAPSED TIME %f" % elapsed
	print "prediction distrib"
	print np.bincount(np.array(tot_pred))
	print "test distrib"
	print np.bincount(np.array(label_test))


	print "TEST F-Measure: %f" % f1_score(label_test, tot_pred, average='weighted')
	print f1_score(label_test, tot_pred, average=None)
	print "TEST Accuracy: %f" % accuracy_score(label_test, tot_pred)
	#print confusion_matrix(label_test, tot_pred)
	print "==========================================="
	sys.stdout.flush()	
	return accuracy_score(label_test, tot_pred)



def getPrediction(x_rnn, nunits, nlayer, nclasses, model_type):
	features_learnt = None
	prediction = None
	vec_feat = None
	ts_shape = x_rnn.get_shape()
	n_timestamps = ts_shape[1].value
	dim_bands = ts_shape[2].value
	
	if model_type == "CNN1D": 	
		x_cnn1D = tf.reshape(x_rnn, [-1, 1, n_timestamps, dim_bands] )
		vec_feat, _ = CNN1D(x_cnn1D, nunits)
	elif model_type == "RNN":
		vec_feat = Rnn(x_rnn, nunits, nlayer, n_timestamps)
	
	#weights = tf.Variable(tf.random_normal([nunits, nclasses], stddev=0.35), name="weights")
	#bias = tf.Variable(tf.random_normal([nclasses], stddev=0.35), name="weights_b")
	#pred = tf.matmul(vec_feat,weights) + bias
	pred = tf.layers.dense(vec_feat, nclasses, activation=None)
	return [pred, vec_feat]



def getLabelFormat(Y):
	new_Y = []
	vals = np.unique(np.array(Y))	
	for el in Y:
		t = np.zeros(len(vals))
		t[el] = 1.0
		#t[hash_conv[el]] = 1.0
		new_Y.append(t)
	return np.array(new_Y)

	
def getRNNFormat(X, n_timetamps):
    #print X.shape
    new_X = []
    for row in X:
        new_X.append( np.split(row, n_timetamps) )
    return np.array(new_X)




def Rnn(x, nunits, nlayer, n_timetamps):
	#PLACEHOLDERS + WEIGHT & BIAS DEF
    #Processing input tensor
	x = tf.unstack(x,n_timetamps,1)
	
	#NETWORK DEF
	#MORE THEN ONE LAYER: list of LSTMcell,nunits hidden units each, for each layer
	if nlayer>1:
		cells=[]
		for _ in range(nlayer):
			cell = rnn.LSTMCell(nunits)
			#cell = rnn.GRUCell(nunits)
			cells.append(cell)
		cell = tf.contrib.rnn.MultiRNNCell(cells)
    #SIGNLE LAYER: single GRUCell, nunits hidden units each
	else:
		cell = rnn.LSTMCell(nunits)
		#cell = rnn.GRUCell(nunits)
	outputs,_=rnn.static_rnn(cell, x, dtype="float32")
	return outputs[-1]



def CNN1D(x, nunits):
	conv1 = tf.layers.conv2d(
	      inputs=x,
	      filters=nunits/2,
	      kernel_size=[1, 7],
	      padding="valid",
	      activation=tf.nn.relu)
	
	conv1 = tf.layers.batch_normalization(conv1)
	print conv1.get_shape()
	
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 2], strides=2)
	
	conv2 = tf.layers.conv2d(
	      inputs=pool1,
	      filters=nunits,
	      kernel_size=[1, 3],
	      padding="valid",
	      activation=tf.nn.relu)
	
	conv2 = tf.layers.batch_normalization(conv2)
	
	print conv2.get_shape()
	
	conv3 = tf.layers.conv2d(
	      inputs=conv2,
	      filters=nunits,
	      kernel_size=[1, 3],
	      padding="same",
		  #padding="valid",
	      activation=tf.nn.relu)
	
	conv3 = tf.layers.batch_normalization(conv3)
	
	conv3 = tf.concat([conv2,conv3],3)
	
	print conv3.get_shape()
	
	conv4 = tf.layers.conv2d(
	      inputs=conv3,
	      filters=nunits,
	      kernel_size=[1, 1],
	      padding="valid",
	      activation=tf.nn.relu)
	
	conv4 = tf.layers.batch_normalization(conv4)
	
	print conv4.get_shape()
	
	conv4_shape = conv4.get_shape()
	cnn = tf.reduce_mean(conv4, [1,2])
	print cnn.get_shape()
	tensor_shape = cnn.get_shape()
	return cnn, tensor_shape[1].value
	
	
	
	
if __name__ == "__main__":
	#Model parameters
	nunits = 512
	batchsz = 32
	hm_epochs = 200
	n_levels_lstm = 1


	#Data INformation
	n_timestamps = 34#40
	n_dims = 16

	ts_train = np.load(sys.argv[1])

	label_train = np.load(sys.argv[2])

	nclasses = len(np.unique(label_train))

	print type(label_train)
	print "nclasses %d" % nclasses

	label_train = label_train.astype('int')

	ts_test = np.load(sys.argv[3])

	label_test = np.load(sys.argv[4])
	label_test = label_test.astype("int")
	
	split_numb = int(sys.argv[5])
	model_type = sys.argv[6]
	
	
	x_rnn = tf.placeholder("float",[None,n_timestamps,n_dims],name="x_rnn")
	y = tf.placeholder("float",[None,nclasses],name="y")
	l_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
	
	sess = tf.InteractiveSession()
	
	pred, features = getPrediction(x_rnn, nunits, n_levels_lstm, nclasses, model_type)
	
	testPrediction = tf.argmax(pred, 1, name="prediction")

	loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred)  )

	optimizer = None
	
	optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(loss)

	
	correct = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct,tf.float64))

	tf.global_variables_initializer().run()

	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()

	classes = label_train - 1
	rnn_data_train = getRNNFormat(ts_train, n_timestamps)
	train_y = getLabelFormat(classes)
	
	rnn_data_test = getRNNFormat(ts_test, n_timestamps)
	
	classes_test = label_test - 1
	test_y = getLabelFormat(classes_test)

	iterations = rnn_data_train.shape[0] / batchsz

	if rnn_data_train.shape[0] % batchsz != 0:
	    iterations+=1

	best_loss = sys.float_info.max




	for e in range(hm_epochs):
		lossi = 0
		accS = 0

		rnn_data_train, train_y = shuffle(rnn_data_train, train_y, random_state=0)
		print "shuffle DONE"

		start = time.time()
		for ibatch in range(iterations):
		#for ibatch in range(10):
			#BATCH_X BATCH_Y: i-th batches of train_indices_x and train_y
			batch_rnn_x, batch_y = getBatch(rnn_data_train, train_y, ibatch, batchsz)

			acc,_,err = sess.run( [accuracy,optimizer,loss],feed_dict={x_rnn:batch_rnn_x, y:batch_y, l_rate:0.0002})		
			lossi+=err
			accS+=acc

			del batch_rnn_x
			del batch_y

		end = time.time()
		elapsed = end - start	
		print "Epoch:",e,"Train loss:",lossi/iterations,"| accuracy:",accS/iterations," | TIME: ", elapsed

		c_loss = lossi/iterations

		if c_loss < best_loss:
			save_path = saver.save(sess, "models/model_"+str(split_numb))
			print("Model saved in path: %s" % save_path)
			best_loss = c_loss
		'''	
		'''
		test_acc = checkTest(rnn_data_test, 1024, classes_test)

	#np.save(outputFileName, np.array(tot_pred))

		
	