import datatools


indir = 'data_test/imgs'
in_csv = 'data_test/data_CP_bboxHW.csv'
feature_header = 'CP'

## Sample input csv to bboxcsv_to_minmaxcsv :
## FN : Filename
## RBP : Feature label
## bboxH : Bounding box height
## bboxW : Bounding box width
#                                                   FN         RBP  dist  bboxH  bboxW
# 0     qefmascom_TopSideImage_2017-06-22-12_41_54.jpg  [331, 368]   478  239.0   95.6
# 1     tjfmascom_TopSideImage_2017-06-28-12_47_39.jpg  [306, 445]   413  206.5   82.6
# 2     kjfmascom_TopSideImage_2017-04-19-16_35_03.jpg  [410, 395]   432  216.0   86.4
# 3  runbsfmascom_TopSideImage_2017-07-04-02_41_08.jpg  [306, 396]   469  234.5   93.8
# 4    sddfmascom_TopSideImage_2017-06-23-17_24_25.jpg  [309, 372]   440  220.0   88.0

out_csv = in_csv.replace('.csv','_xyminmax.csv')
df_out, out_csv = datatools.od.bboxcsv_to_minmaxcsv(indir, in_csv, feature_header=feature_header, ptformat='Center', out_csv=out_csv, PRINT = False)
