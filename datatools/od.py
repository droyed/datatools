import numpy as np
import os
import pandas as pd
import imagesize
from PIL import Image
import shutil
import xml.etree.ElementTree as ET
import ast
import cv2


def iterinfo(fpaths, iterID):
    fpath = fpaths[iterID]
    fname = os.path.basename(fpath)
    print('------------ '+str(iterID)+'/'+str(len(fpaths)-1)+' : '+fname)

def mkdir(dirn):
    if not os.path.exists(dirn):
        os.makedirs(dirn)

# Freshly create output dir
def newmkdir(P):
    if os.path.isdir(P):
        shutil.rmtree(P)    
    mkdir(P)

# Show UINT8 array as an image on CV2 figure window    
def image_show(window_name,image):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name,image)

# Display window
def DisplayWindow():
    while(1):
        k = cv2.waitKey(33)
        if k==27:    # Space key to stop
             cv2.destroyAllWindows()
             break
        elif k & 0xFF == ord('q') :
            cv2.destroyAllWindows()
        else:
            continue
    return

def rint(a):
    return int(np.round(a))

def create_xml(template_file, info, outxml):
    filepath = info['filepath']
    label = info['label']
    bbox = info['bbox']

    ET.parse(template_file)
    tree = ET.parse(template_file)
    root = tree.getroot()
    
    # Set fname, fpath
    fname = os.path.basename(filepath)
    root[1].text = fname
    root[2].text = filepath
    
    # Get img data
    im = Image.open(filepath)
    shp = np.array(im).shape
    
    # Set img size
    root[4][0].text = str(shp[1])
    root[4][1].text = str(shp[0])
    root[4][2].text = str(shp[2])
    
    # Set label
    root[6][0].text = label
    
    # Set bbox
    for i in range(4):
        root[6][4][i].text = str(bbox[i])
    
    tree.write(outxml)
    return

def setup_dict_for_xml_conversion(d, label, xml_template, outxml, filepath_header='filepath'):
    info = {}
    info['filepath'] = d[filepath_header]
    info['label'] = label
    info['bbox'] = d['xmin'], d['ymin'], d['xmax'], d['ymax']
    create_xml(xml_template, info, outxml=outxml)
    return

def draw_bbox(fpath, info, label=None, line_thickness = 2, SHOW=False):
    ## info = (xmin, ymin, xmax, ymax)
    im_draw = cv2.imread(fpath)
    
    if isinstance(info, dict):
        feature_xmin, feature_ymin, feature_xmax, feature_ymax = info['xmin'], info['ymin'], info['xmax'], info['ymax']
    elif isinstance(info, list) or isinstance(info, tuple):
        feature_xmin, feature_ymin, feature_xmax, feature_ymax = info
    else:
        raise Exception('Input form at not recognized!')
    
    cv2.line(im_draw,(feature_xmin, feature_ymin),(feature_xmin, feature_ymax),(255,0,0),line_thickness)
    cv2.line(im_draw,(feature_xmin, feature_ymax),(feature_xmax, feature_ymax),(255,0,0),line_thickness)
    cv2.line(im_draw,(feature_xmax, feature_ymax),(feature_xmax, feature_ymin),(255,0,0),line_thickness)
    cv2.line(im_draw,(feature_xmax, feature_ymin),(feature_xmin, feature_ymin),(255,0,0),line_thickness)
    cv2.putText(im_draw, label, (feature_xmax, feature_ymax),cv2.LINE_AA,1,(0,0,0),2)
    
    if SHOW:
        image_show('im_draw', im_draw)
        DisplayWindow()

    return im_draw

def get_bbox_from_featurept_per_image(pt, bboxH, bboxW, imgH, imgW, ptformat, PRINT=False, RINT=True):    
    if bboxW<=1.0:
        bboxW = rint(bboxW*imgW)
    
    if bboxH<=1.0:
        bboxH = rint(bboxH*imgH)

    if RINT:
        bboxW = rint(bboxW)
        bboxH = rint(bboxH)
            
    if ptformat == 'Center':
        xmin = pt[0] - bboxW//2
        xmax = pt[0] + bboxW//2
        ymin = pt[1] - bboxH//2
        ymax = pt[1] + bboxH//2
    elif ptformat == 'MidTLBL':
        xmin = pt[0]
        xmax = pt[0] + bboxW
        ymin = pt[1] - bboxH//2
        ymax = pt[1] + bboxH//2
    elif ptformat == 'MidTRBR':
        xmin = pt[0] - bboxW
        xmax = pt[0]
        ymin = pt[1] - bboxH//2
        ymax = pt[1] + bboxH//2
    elif ptformat == 'TL':
        xmin = pt[0]
        xmax = pt[0] + bboxW
        ymin = pt[1]
        ymax = pt[1] + bboxH
    elif ptformat == 'TR':
        xmin = pt[0] - bboxW
        xmax = pt[0]
        ymin = pt[1]
        ymax = pt[1] + bboxH
    else:
        raise Exception('Wrong value for ptformat. Accepted values are : "MidTLBL", "MidTRBR", "TL", "TR"')

    # Fix out-of-bounds
    xmin = max(0, xmin)
    xmax = min(imgW, xmax)
    ymin = max(0, ymin)
    ymax = min(imgH, ymax)
    
    bbox = {
            'xmin':xmin,
            'xmax':xmax,
            'ymin':ymin,
            'ymax':ymax,
            }
    
    if PRINT:
        print('pt : '+str(pt))
        print('bboxW : '+str(bboxW))
        print('bboxH : '+str(bboxH))
        print('bbox : ' + str(bbox))
    return bbox

def bboxcsv_to_minmaxcsv(indir, in_csv, feature_header, ptformat, out_csv, PRINT = False):
    ## indir : input dir of images
    ## in_csv : Input csv with data of feature points
    ## feature_header : header for the feature pt

    ## 'FN' : filename header
    ## 'bboxH' : bounding box header for height
    ## 'bboxW' : bounding box header for width

    ## Sample input csv :    
    #                                                   FN         RBP  dist  bboxH  bboxW
    # 0     qefmascom_TopSideImage_2017-06-22-12_41_54.jpg  [331, 368]   478  239.0   95.6
    # 1     tjfmascom_TopSideImage_2017-06-28-12_47_39.jpg  [306, 445]   413  206.5   82.6
    # 2     kjfmascom_TopSideImage_2017-04-19-16_35_03.jpg  [410, 395]   432  216.0   86.4
    # 3  runbsfmascom_TopSideImage_2017-07-04-02_41_08.jpg  [306, 396]   469  234.5   93.8
    # 4    sddfmascom_TopSideImage_2017-06-23-17_24_25.jpg  [309, 372]   440  220.0   88.0
    
    # Read in csv as df
    df_in = pd.read_csv(in_csv)
    
    #print('df_in head :')
    #print(df_in.head())
    
    bboxs = []
    for index, row in df_in.iterrows():    
        d = dict(row)    
        pt = tuple(ast.literal_eval(d[feature_header])) 
        bboxH = d['bboxH']
        bboxW = d['bboxW']
        fp = os.path.join(indir, d['FN'])
        imgW, imgH = imagesize.get(fp)
        bbox = get_bbox_from_featurept_per_image(pt, bboxH, bboxW, imgH, imgW, ptformat=ptformat, PRINT=PRINT, RINT=True)
        #info_i = {'FN':d['FN'], 'FP':fp}
        info_i = {'FN':d['FN']}        
        info_i.update(bbox)
        bboxs.append(info_i)
        
    df_out = pd.DataFrame(bboxs)
    
    #print('df_out head :')
    #print(df_out.head())

    df_out.to_csv(out_csv, index=False)
    print('Output csv of bbox info saved at - '+out_csv)
    return df_out, out_csv

def minmaxcsv_to_setup(indir, in_csv, label, outdir_imgsxmls, outdir_debug, NUM_DEBUG=None):    
    ## 'FN' : filename header
    ## 'FP' : filepath header

    ## Sample input csv :    
    #                                                   FN                                                 FP  xmin  xmax  ymin  ymax
    # 0     qefmascom_TopSideImage_2017-06-22-12_41_54.jpg  /home/diva/pprojs/data_xml/SinglePaper-Portrai...   331   427   249   487
    # 1     tjfmascom_TopSideImage_2017-06-28-12_47_39.jpg  /home/diva/pprojs/data_xml/SinglePaper-Portrai...   306   389   342   548
    # 2     kjfmascom_TopSideImage_2017-04-19-16_35_03.jpg  /home/diva/pprojs/data_xml/SinglePaper-Portrai...   410   496   287   503
    # 3  runbsfmascom_TopSideImage_2017-07-04-02_41_08.jpg  /home/diva/pprojs/data_xml/SinglePaper-Portrai...   306   400   279   513
    # 4    sddfmascom_TopSideImage_2017-06-23-17_24_25.jpg  /home/diva/pprojs/data_xml/SinglePaper-Portrai...   309   397   262   482

    # Get template path
    xml_template = os.path.join(os.path.dirname(__file__), 'od_template.xml')
    
    # Read in csv as df
    df_out = pd.read_csv(in_csv)

    newmkdir(outdir_imgsxmls)
    newmkdir(outdir_debug)
    
    if NUM_DEBUG is not None:
        DEBUG_IDS = np.random.choice(len(df_out), min(len(df_out), NUM_DEBUG), replace=False)
    else:
        DEBUG_IDS = np.arange(len(df_out))
    
    ## Create xml files, debug output images
    for iterID,(index, row) in enumerate(df_out.iterrows()):
        iterinfo(df_out['FN'].values, iterID)
        d = dict(row)
        
        # Copy over the input images directly into output directory
        d['FP'] = os.path.join(indir, d['FN'])
        src = d['FP']
        dst = os.path.join(outdir_imgsxmls, os.path.basename(src))
        shutil.copyfile(src, dst)
        # Create xmls
        xml_path = os.path.join(outdir_imgsxmls, os.path.splitext(d['FN'])[0] + '.xml')
        setup_dict_for_xml_conversion(d, label, xml_template=xml_template, outxml=xml_path, filepath_header='FP')
    
        if iterID in DEBUG_IDS:
            # Create debug imgs with bbox marked 
            imd = draw_bbox(d['FP'], info=d, label=label, SHOW=False)    
            cv2.imwrite(os.path.join(outdir_debug, d['FN']), imd)
     
    print('Output dir of images and xmls saved at : '+outdir_imgsxmls)
    print('Output dir of marked bounding boxes saved at : '+outdir_debug)
    return

def features_to_minmaxcsv(incsv, feature_headers, outcsv):
    df = pd.read_csv(incsv)
    
    out = []
    for index,row in df.iterrows():
        d = dict(row)
        
        a = np.vstack([ast.literal_eval(d[fh]) for fh in feature_headers])
        xmin,ymin = a.min(0)
        xmax,ymax = a.max(0)
        
        out_i = {}
        out_i['FN'] = d['FN']
        out_i['xmin'] = xmin
        out_i['ymin'] = ymin
        out_i['xmax'] = xmax
        out_i['ymax'] = ymax
        out.append(out_i)
    
    pd.DataFrame(out).to_csv(outcsv, index=False)
    print('Output csv with min-max bounds saved at : '+outcsv)
    return
