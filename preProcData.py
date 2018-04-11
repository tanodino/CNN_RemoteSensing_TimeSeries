import glob
import gdal
import sys
import numpy as np

hash_class = {}
hash_class[0] = 0
hash_class[14] = 0

hash_class[1] = 1
hash_class[2] = 1
hash_class[5] = 1

hash_class[3] = 2
hash_class[6] = 2
hash_class[7] = 2
hash_class[13] = 2

hash_class[10] = 3
hash_class[12] = 3

hash_class[4] = 4
hash_class[8] = 5
hash_class[9] = 5
hash_class[11] = 6

hash_class[15] = 7
hash_class[16] = 8
hash_class[17] = 9



def addNDVI(img):
	red = img[:,:,0]
	nir = img[:,:,-1]
	ndvi = ( ((nir - red) / (nir + red)) + 1) / 2
	ndvi = np.reshape(ndvi, (ndvi.shape[0], ndvi.shape[1], 1))
	return np.concatenate((img,ndvi), axis=2)
	

def computeNDVI(vhsr_train):
	new_data = []
	for img in vhsr_train:
		new_data.append( addNDVI(img) )
	return np.array(new_data)


def obtainVHSRData(vhsr_fileName, valid_pixels):
	# I know that the minimum value of the BAND is 1
	min_val = 1
	scaling_factor = 5
	buffer_window = 2
	data_vhsr = []
	for k in range(len(valid_pixels)):
		data_vhsr.append([])
	
	ds = gdal.Open(vhsr_fileName)
	for band in range(1, ds.RasterCount+1):
		print "\tanalyze band %d" % band
		t_matrix = ds.GetRasterBand(band).ReadAsArray()
		max_val = np.amax(t_matrix)
		t_matrix = (t_matrix.astype('float32') - min_val) / (max_val - min_val)
		
		for k in range(len(valid_pixels)):
			i, j = valid_pixels[k]
			begin_i = (i-buffer_window) * scaling_factor
			begin_i = int(begin_i)
			end_i =  (i+1+buffer_window) * scaling_factor
			end_i = int(end_i)
			begin_j = (j-buffer_window) * scaling_factor
			begin_j = int(begin_j)
			end_j = (j+1+buffer_window) * scaling_factor
			end_j = int(end_j)
			data_vhsr[k].append( t_matrix[begin_i:end_i, begin_j:end_j] )
	ds = None
	return np.array(data_vhsr)
	

def getMinMax(mnew):
	max_ = np.amax(mnew)
	#min_ = np.amin(mnew)
	min_ = np.amin(mnew[mnew != np.amin(mnew)])
	return min_,max_

def normalizeB(mnew):
	max_ = np.amax(mnew)
	#min_ = np.amin(mnew)
	min_ = np.amin(mnew[mnew != np.amin(mnew)])
	
	if max_ == min_:
		return mnew * 0
	else:
		return (mnew.astype('float32') - min_) / (max_ - min_)


def normalize(mnew, min_, max_):
	return (mnew.astype('float32') - min_) / (max_ - min_)

def readGT(gt_fileName, obj_fileName, NO_DATA_GT):
	ds = gdal.Open(gt_fileName)
	gt_matrix = ds.GetRasterBand(1).ReadAsArray()
	ds = None
	
	ds = gdal.Open(obj_fileName)
	print "open %s" % obj_fileName
	obj_matrix = ds.GetRasterBand(1).ReadAsArray()
	ds = None	
	
	ds_label = []
	for i in range(gt_matrix.shape[0]):
		for j in range(gt_matrix.shape[1]):
			if hash_class[ gt_matrix[i][j] ] != NO_DATA_GT:
				ds_label.append( [i,j, hash_class[ gt_matrix[i][j]], obj_matrix[i][j]] )
	ds_label = np.array(ds_label)	
	return ds_label


def readBand(fileName, valid_pixels, NO_DATA_VALUE):
	
	ds = gdal.Open(fileName)
	band_data = []
	gmax = None
	gmin = None
	count = 0
	for k in range(len(valid_pixels)):
		band_data.append([])
	
	for band in range(1,ds.RasterCount+1):
		t_matrix = ds.GetRasterBand(band).ReadAsArray()
		min_, max_ = getMinMax(t_matrix)
		#print "max_ %d" % max_
		#print "min_ %d" % min_
		if gmax is None:
			gmax = max_
			gmin = min_
		else:
			gmax = max(max_, gmax)
			gmin = min(min_, gmin)
		
		for k in range(len(valid_pixels)):
			i,j = valid_pixels[k]
			i = int(i)
			j = int(j)
			band_data[k].append(t_matrix[i][j])	
	
	band_data = np.array(band_data)	
	no_valid = []
	for i in range(len(band_data)):
		t = np.array(band_data[i])
		c = len(t[ np.where( t < -10000 ) ])
		if c != 0:
			no_valid.append(i)
			
	no_valid = np.array(no_valid)

	ds = None
	
	band_data = normalize(band_data, gmin, gmax)
	return band_data, no_valid

directory = "."
gt_fileName = directory+"/GT/gt_classes.tif"
obj_fileName = directory+"/GT/gt_obj.tif"
tsdata_fileDir = directory+"/SERIES"
vhsr_fileName = directory+"/VHSR/IMG_S6PAN_2016031236520790CP_R1C1_CUT2.TIF"
NO_DATA_VALUE = -9999
NO_DATA_GT = 0

print "READ CLASS and OBJ fileS"
ds_label = readGT(gt_fileName, obj_fileName, NO_DATA_GT)

valid_pixels = ds_label[:,0:2]


print "VHSR DATA"
vhsr_data = obtainVHSRData(vhsr_fileName, valid_pixels)

print vhsr_data.shape

#exit()

print "READ TIMESERIES DATA"
files = glob.glob(tsdata_fileDir+"/*.tif")

hash_bands = {}
ts_length = None
g_no_valid = np.array([])

for fileName in files:
	print "analyze file: %s" % fileName
	sys.stdout.flush()
	band_ts, local_no_valid = readBand(fileName, valid_pixels, NO_DATA_VALUE)
	g_no_valid = np.append(g_no_valid, local_no_valid)
	hash_bands[fileName] = band_ts
	ts_length = band_ts.shape[1]
	print "local_no_valid %d" % len(local_no_valid)

g_no_valid = np.unique(g_no_valid)
print "NO VALID PIXELS: %d" % len(g_no_valid)


total_ids = np.array(range(len(ds_label)))
valid_ids = np.setdiff1d(total_ids, g_no_valid)
valid_ids = np.sort(valid_ids)


np.save("ds_label.npy",ds_label[valid_ids])

vhsr_data = vhsr_data[valid_ids]
vhsr_data = np.swapaxes(vhsr_data,1,3)
vhsr_data = computeNDVI(vhsr_data)

np.save("vhsr_data.npy", vhsr_data)

ts_data = None	
for i in range(ts_length):
	for band_name in hash_bands:
		if ts_data is None:
			ts_data = hash_bands[band_name][:,i]
		else:
			ts_data = np.column_stack((ts_data, hash_bands[band_name][:,i]))

np.save("ts_data.npy",ts_data[valid_ids])		
	


	

