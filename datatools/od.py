import numpy as np
import os
import pandas as pd
import imagesize
from PIL import Image
import shutil
import xml.etree.ElementTree as ET
import ast
import cv2
import copy


# Get template path
xml_template = os.path.join(os.path.dirname(__file__), 'od_template.xml')


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

    ## Sample input csv (in_csv, CP is feature header) :
    #
    #           FN          CP  bboxH  bboxW
    # 0  00003.jpg   (165, 76)    218     87
    # 1  00032.jpg   (128, 67)    243     97
    # 2  00055.jpg   (242, 81)    219     87
    # 3  00148.jpg   (251, 93)    247     98
    # 4  00175.jpg  (304, 201)    219     87
    #
    ## Sample input csv (out_csv, CP is feature header) :
    #
    #           FN  xmin  xmax  ymin  ymax
    # 0  00003.jpg   122   208     0   185
    # 1  00032.jpg    80   176     0   188
    # 2  00055.jpg   198   286     0   190
    # 3  00148.jpg   202   300     0   216
    # 4  00175.jpg   260   348    92   310

    # Read in csv as df
    df_in = pd.read_csv(in_csv)
    
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

    df_out.to_csv(out_csv, index=False)
    print('Output csv of bbox info saved at - '+out_csv)
    return df_out, out_csv

def minmaxcsv_to_setup(indir, in_csv, label, outdir_imgsxmls, outdir_debug, NUM_DEBUG=None):    
    ## 'FN' : filename header
    ## 'FP' : filepath header

    ## Sample input csv :    
    #           FN  xmin  ymin  xmax  ymax
    # 0  00003.jpg   102     4   346   273
    # 1  00032.jpg    73    27   244   340
    # 2  00055.jpg   148    19   293   321
    # 3  00148.jpg   220    31   466   403
    # 4  00175.jpg   224    61   532   439
        
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
    
# Convert/evaluate strings from csv headers
def ast_literal_eval_inplace(df, headers):
    for h in headers:
        df[h] = [ast.literal_eval(i) for i in df[h]]
    return df

# Edit object element in template xml for tf object detection     
def edit_object_element(p0, new_object_name, new_object_bbox, make_copy=False):
    # Edit object name and bounding box into xml file
    if make_copy:
        p = copy.deepcopy(p0)
    else:
        p = p0
    k1 = p.find('name')
    k1.text = new_object_name
    
    k2 = p.find('bndbox')
    for k,v in new_object_bbox.items():
        k2_1 = k2.find(k)
        k2_1.text = str(v)
    return p

# Create formatted tf object detection xml file from given parameters
def format_xml(filename, filepath, objects, out_xml_fpath):
    ## Syntax :
    # filename : 'fname'
    # filename : 'fpath'
    # objects = {
    #     'cat':{'xmin':988, 'ymin':20, 'xmax':70, 'ymax':130},
    #     'dog':{'xmin':288, 'ymin':50, 'xmax':20, 'ymax':430},
    #     'pig':{'xmin':588, 'ymin':70, 'xmax':90, 'ymax':230},
    #     }
    # out_xml_fpath : 'out.xml'

    tree = ET.parse(xml_template)
    root = tree.getroot()
    
    root.find('filename').text = filename
    root.find('path').text = filepath
    
    p = root.find('object')
    for iterID,(k,v) in enumerate(objects.items()):
        make_copy = iterID>0
        p_k = edit_object_element(p, new_object_name=k, new_object_bbox=v, make_copy=make_copy)
        if make_copy:
            root.append(p_k)
    
    tree.write(out_xml_fpath)
    return

# Draw bounding box in a debug image 
def draw_bbox_inimg(im_draw, info, label=None, line_thickness = 2, SHOW=False):    
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

# Input a csv with min-max bounding box info across one or more features and output directory of images and xmls
def minmaxcsv_to_setup_multiple(indir, in_csv, feature_labels, outdir_imgsxmls, outdir_debug, NUM_DEBUG=None):
    ## Syntax :
    # indir : input dir of images
    # in_csv : input csv with format :
    #   
    #           FN                 cat                  dog                  pig
    # 0  00003.jpg  (108, 4, 200, 150)  (218, 23, 290, 120)   (108, 4, 200, 150)
    # 1  00032.jpg  (38, 55, 160, 120)  (118, 53, 190, 220)  (128, 34, 192, 250)
    #
    # feature_labels : ['cat', 'dog', 'pig']
    # outdir_imgsxmls : output dir to save images and xmls
    # outdir_debug : output dir to save images with marked bounding boxes
    # NUM_DEBUG : number of debug images to be saved with marked bounding boxes

    newmkdir(outdir_imgsxmls)
    newmkdir(outdir_debug)
    
    df = pd.read_csv(in_csv)
    ast_literal_eval_inplace(df, feature_labels)
    
    if NUM_DEBUG is not None:
        DEBUG_IDS = np.random.choice(len(df), min(len(df), NUM_DEBUG), replace=False)
    else:
        DEBUG_IDS = np.arange(len(df))
    
    for iterID,(index,row) in enumerate(df.iterrows()):
        iterinfo(df.FN.values, iterID)
        d = dict(row)
        dict1 = {key:dict(zip(['xmin', 'ymin', 'xmax', 'ymax'], d[key])) for key in feature_labels}    
        out_xml = os.path.join(outdir_imgsxmls, os.path.splitext(d['FN'])[0]+'.xml')
        
        # Copy over the input images directly into output directory
        fn = d['FN']
        src = os.path.join(indir, fn)
        dst = os.path.join(outdir_imgsxmls, fn)
        shutil.copyfile(src, dst)        
        format_xml(filename=fn, filepath=dst, objects=dict1, out_xml_fpath=out_xml)
        
        # Create debug imgs with bbox marked 
        if iterID in DEBUG_IDS:
            im_draw = cv2.imread(src)
            for k,v in dict1.items():
                draw_bbox_inimg(im_draw, info=v, label=k, SHOW=False)    
            cv2.imwrite(os.path.join(outdir_debug, d['FN']), im_draw)
            
    return

# Convert a df with feature point headers to a single header defined by feature extents
def features_to_minmaxdf(df, feature_headers, out_feature_header):
    # Input csv :
    #          FN          TL          TR           BL
    # 0  0001.jpg  (249, 350)  (783, 358)  (261, 1050)
    # 1  0002.jpg  (260, 340)  (781, 322)  (301, 1019)
    #    
    # Output csv :
    #          FN                    FUP
    # 0  0001.jpg   [249, 350, 783, 1050]
    # 1  0002.jpg.  [260, 322, 781, 1019]
        
    out = []
    for index,row in df.iterrows():
        d = dict(row)
        
        a = np.vstack([ast.literal_eval(d[fh]) for fh in feature_headers])
        xmin,ymin = a.min(0)
        xmax,ymax = a.max(0)
        
        out_i = {}
        out_i['FN'] = d['FN']
        out_i[out_feature_header] = [xmin, ymin, xmax, ymax]
        out.append(out_i)
    
    df_out = pd.DataFrame(out)
    return df_out

