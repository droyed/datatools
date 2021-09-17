import datatools


indir = 'data_test/imgs'
incsv = 'data_test/data_3P.csv'
label = 'KANG'
feature_headers = ['TL', 'TR', 'BL']

## Sample input csv to minmaxcsv_to_setup :
## FN : Filename
## TL, TR, BL are the three features
#
#           FN         TL         TR          BL
# 0  00003.jpg   (108, 4)  (346, 13)  (102, 273)
# 1  00032.jpg   (81, 37)  (244, 27)   (73, 340)
# 2  00055.jpg  (190, 19)  (293, 21)  (148, 321)
# 3  00148.jpg  (228, 36)  (466, 31)  (220, 403)
# 4  00175.jpg  (278, 77)  (532, 61)  (224, 439)

outcsv = incsv.replace('.csv','_minmax.csv')
datatools.od.features_to_minmaxcsv(incsv, feature_headers, outcsv)
datatools.od.minmaxcsv_to_setup(indir, in_csv=outcsv, label=label, outdir_imgsxmls=indir + '_imgs_xmls', outdir_debug=indir + '_bboxmarkedimgs', NUM_DEBUG=100)
