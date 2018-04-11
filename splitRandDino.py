import numpy as np
import random
import os
from datetime import datetime

def writeSplit(idx, data, outFileName ):
	fold = []
	for i in idx:
		fold.append( data[i] )
	fold = np.array(fold)
	np.save(outFileName, fold )

def getIdxSplit(hash_class, perc_train):
	train_idx = []
	test_idx = []
	dt = datetime.now()
	random.seed( dt.microsecond )
	classes = hash_class.keys()
	
	for cl in classes:
		vals = hash_class[cl].keys()	
		random.shuffle(vals)
		thr = int(len(vals) * perc_train)
		print "%d over %d" % (thr, len(vals))
		for i in range(len(vals)):
			if i < thr:
				train_idx.extend( hash_class[cl][vals[i]] ) 
			else:
				test_idx.extend( hash_class[cl][vals[i]] ) 
	print "============"
	return train_idx, test_idx
		
	

path_in = "data_orig"
path_out = "splits"

if not os.path.exists(path_out):
	os.makedirs(path_out)

ds_label = np.load( path_in+"/ds_label.npy" )
vhsr = np.load( path_in+"/vhsr_data.npy" )
timeSeries = np.load( path_in+"/ts_data.npy" )

perc_train = 0.3
#split_id = 0


hash_class = {}
pid = 0
for el in ds_label:
	cl = el[2]
	obj_id = el[3]
	if cl not in hash_class:
		hash_class[cl] = {}
		
	if obj_id not in hash_class[cl]:
		hash_class[cl][obj_id] = []
	hash_class[cl][obj_id].append( pid )	
	pid = pid + 1

print hash_class.keys()

if not os.path.exists(path_out+"/TimeSeries"):
	os.makedirs(path_out+"/TimeSeries")

if not os.path.exists(path_out+"/VHSR"):
	os.makedirs(path_out+"/VHSR/")
	
if not os.path.exists(path_out+"/ground_truth"):
	os.makedirs(path_out+"/ground_truth/")
	
	
for split_id in range(5):
	train_idx, test_idx = getIdxSplit(hash_class, perc_train)
	
	
	#write time series splits
	outFileTrain = path_out+"/TimeSeries/train_x"+str(split_id)+"_"+str(int(perc_train*100))+".npy"
	writeSplit(train_idx, timeSeries, outFileTrain )

	outFileTest = path_out+"/TimeSeries/test_x"+str(split_id)+"_"+str(int(perc_train*100))+".npy"
	writeSplit(test_idx, timeSeries, outFileTest )


	#write vhsr splits
	outFileTrain = path_out+"/VHSR/train_x"+str(split_id)+"_"+str(int(perc_train*100))+".npy"
	writeSplit(train_idx, vhsr, outFileTrain )

	outFileTest = path_out+"/VHSR/test_x"+str(split_id)+"_"+str(int(perc_train*100))+".npy"
	writeSplit(test_idx, vhsr, outFileTest )
	
	#write ground_truth
	outFileTrain = path_out+"/ground_truth/train_y"+str(split_id)+"_"+str(int(perc_train*100))+".npy"
	np.save(outFileTrain, np.array( ds_label[train_idx, 2 ] ))


	outFileTest = path_out+"/ground_truth/test_y"+str(split_id)+"_"+str(int(perc_train*100))+".npy"
	np.save(outFileTest, np.array( ds_label[test_idx, 2 ] ))



