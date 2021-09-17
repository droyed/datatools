#!/bin/bash

echo "==> (1) Demo for s1_to_s2 ..."
python start1_to_start2.py 1
python start1_to_start2.py 2
python start1_to_start2.py 3

echo "==> (2) Demo for bboxcsv_to_minmaxcsv ..."
python demo_bboxcsv_to_minmaxcsv.py

echo "==> (3) Demo for features_to_minmaxcsv ..."
python demo_features_to_minmaxcsv_to_setup.py

echo "==> (4) Demo for demo_minmaxcsv_to_setup ..."
python demo_minmaxcsv_to_setup.py

exit 0
