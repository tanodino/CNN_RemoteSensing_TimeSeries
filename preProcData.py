import glob
import gdal
import sys
import numpy as np


def getMinMax(mnew):
	max_ = np.amax(mnew)
	min_ = np.amin(mnew)
	#min_ = np.amin(mnew[mnew != np.amin(mnew)])
	return min_,max_

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
			if gt_matrix[i][j]  != NO_DATA_GT:
				ds_label.append( [i,j, gt_matrix[i][j], obj_matrix[i][j]] )
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
		c = len(t[ np.where( t == NO_DATA_VALUE ) ])
		if c != 0:
			no_valid.append(i)
			
	no_valid = np.array(no_valid)

	ds = None
	
	band_data = normalize(band_data, gmin, gmax)
	return band_data, no_valid



if __name__ == "__main__":
	directory = "../Reunion2017"
	gt_fileName = directory+"/GT/RASTERS/BD_Reunion_V2_CodeN3.tif"
	obj_fileName = directory+"/GT/RASTERS/BD_Reunion_V2_ObjID.tif"
	tsdata_fileDir = directory+"/SERIES"
	NO_DATA_VALUE = -9999
	NO_DATA_GT = 0

	print "READ CLASS and OBJ fileS"
	ds_label = readGT(gt_fileName, obj_fileName, NO_DATA_GT)

	valid_pixels = ds_label[:,0:2]


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

	ts_data = None	
	
	for i in range(ts_length):
		for band_name in hash_bands:
			if ts_data is None:
				ts_data = hash_bands[band_name][:,i]
			else:
				ts_data = np.column_stack((ts_data, hash_bands[band_name][:,i]))

	np.save("ts_data.npy",ts_data[valid_ids])		
	


	

