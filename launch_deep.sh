#!/bin/sh


path_ts="splits/TimeSeries"
path_gt="splits/ground_truth"

#$1: fold_id
#$2: training_perc
#$3: model_type
python method.py $path_ts/train_x$1_$2.npy $path_gt/train_y$1_$2.npy $path_ts/test_x$1_$2.npy $path_gt/test_y$1_$2.npy $1 $3

