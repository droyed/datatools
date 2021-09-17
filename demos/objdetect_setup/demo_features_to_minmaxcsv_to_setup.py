import datatools


indir = 'data_test/imgs'
incsv = 'data_test/data_3P.csv'
label = 'KANG'
feature_headers = ['TL', 'TR', 'BL']

# indir = '/home/diva/pprojs/setup_half_sp3/allimgs_SP_3_70p'
# incsv = '/home/diva/pprojs/setup_half_sp3/data_SP3_featurepts_70p.csv'
# label = 'SP3H'
# feature_headers = ['TP', 'LBP', 'RBP']

outcsv = incsv.replace('.csv','_minmax.csv')
datatools.od.features_to_minmaxcsv(incsv, feature_headers, outcsv)
datatools.od.minmaxcsv_to_setup(indir, in_csv=outcsv, label=label, outdir_imgsxmls=indir + '_imgs_xmls', outdir_debug=indir + '_bboxmarkedimgs', NUM_DEBUG=100)
