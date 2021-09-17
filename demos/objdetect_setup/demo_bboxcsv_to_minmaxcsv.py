import datatools


indir = 'data_test/imgs'
in_csv = 'data_test/data_CP_bboxHW.csv'
feature_header = 'CP'

## Sample input csv to bboxcsv_to_minmaxcsv :
## FN : Filename
## CP : Feature label
## bboxH : Bounding box height
## bboxW : Bounding box width
#
#           FN          CP  bboxH  bboxW
# 0  00003.jpg   (165, 76)    218     87
# 1  00032.jpg   (128, 67)    243     97
# 2  00055.jpg   (242, 81)    219     87
# 3  00148.jpg   (251, 93)    247     98
# 4  00175.jpg  (304, 201)    219     87

out_csv = in_csv.replace('.csv','_xyminmax.csv')
df_out, out_csv = datatools.od.bboxcsv_to_minmaxcsv(indir, in_csv, feature_header=feature_header, ptformat='Center', out_csv=out_csv, PRINT = False)
