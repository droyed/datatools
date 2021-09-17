import datatools


indir = 'data_test/imgs'
in_csv = 'data_test/data_CP_bboxHW_xyminmax.csv'
label = 'KANGEYES'

## Sample input csv to minmaxcsv_to_setup :    
## FN : Filename
## FP : Filepath
## xmin : Bounding box xmin
## xmax : Bounding box xmax
## ymin : Bounding box ymin
## ymax : Bounding box ymax
#
#           FN  xmin  xmax  ymin  ymax
# 0  00003.jpg    96   234    19   133
# 1  00032.jpg    56   200     0   139
# 2  00055.jpg   166   318    13   149
# 3  00148.jpg   139   363     8   178
# 4  00175.jpg   214   394   111   291

datatools.od.minmaxcsv_to_setup(indir, in_csv=in_csv, label=label, outdir_imgsxmls=indir + '_imgs_xmls', outdir_debug=indir + '_bboxmarkedimgs')
