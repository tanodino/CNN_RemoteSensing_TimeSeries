#!/bin/sh

path_vhsr="splits/VHSR"
path_ts="splits/TimeSeries"
path_gt="splits/ground_truth"
path_res="splits/results"

python RnnCnnDualNet.py $path_ts/train_x$1_$2.npy $path_vhsr/train_x$1_$2.npy $path_gt/train_y$1_$2.npy $path_ts/test_x$1_$2.npy $path_vhsr/test_x$1_$2.npy $path_gt/test_y$1_$2.npy $1

