import datatools

indir = 'data_test/imgs'
csv = 'data_test/data_3P_minmax.csv'
feature_labels = ['cat', 'dog', 'pig']

datatools.od.minmaxcsv_to_setup_multiple(
    indir=indir, 
    in_csv=csv, 
    feature_labels=feature_labels, 
    outdir_imgsxmls=indir + '_output_imgsxmls', 
    outdir_debug=indir + '_output_markedbbox',
    NUM_DEBUG=10)
