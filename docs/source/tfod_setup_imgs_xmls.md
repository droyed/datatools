## Tensorflow object detection data setup

### Introduction

Tensorflow object detection workflow needs `tfrecord` files as inputs for training. One of the ways to reach there is with a directory of images and xml files - [Create TensorFlow Records](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#convert-xml-to-record), which is a pretty straight-forward step. The xml files capture the bounding box corresponding to the object in context to be detected in a specific format. Asssuming, images are already setup, the work remains on setting up the xml files and this articles focusses on the same. Specifically, we are assuming that the bounding box or related information is available in csvs and our task is to convert them to xml files.

Let's call the required directory as `dirix` (`dir`ectory of `i`mages and `x`mls) for ease of reference. Let's call the starting point on our data as `start` again for ease of reference.  Let's explore how to go from `start` to `dirix` to generate those xmls. Also, we will showcase samples that are taken from `demo` directories in the repository.

The last checkpoint before setting up `dirix`, needs a dataframe or csv with a format as shown below (let's call it `end`) :

```
          FN  xmin  xmax  ymin  ymax
0  00003.jpg   122   208     0   185
1  00032.jpg    80   176     0   188
2  00055.jpg   198   286     0   190
3  00148.jpg   202   300     0   216

```

It's a dataframe with a row entry for each image with their filenames and `min`, `max` along `x` and `y` corresponding to their bounding boxes.

### Scenarios

Now, this checkpoint `end` could be reached in many different ways depending on `start`. We will propose functions and utility scripts to do so. Let's explore some of the common scenarios we might be in before reaching `end`.

**Scenario #1 :**  `start1` : csv with label points in x-y coordinates -

```
          FN          CP
0  00003.jpg   (165, 76)
1  00032.jpg   (128, 67)
2  00055.jpg   (242, 81)
3  00148.jpg   (251, 93)
```

So, here we have a feature named `CP` and are basically interested in finding a keypoint in an image. Let's call it feature point. The hack here is associating a bounding box around the keypoint with some mapping, so that we could employ object detection to do so. Hence, when inferencing, we will map back to that location with a reverse mapping.

**Scenario #2 :**  `start2` : csv with label points in x-y coordinates and bounding box height and width -

```
          FN          CP  bboxH  bboxW
0  00003.jpg   (165, 76)    218     87
1  00032.jpg   (128, 67)    243     97
2  00055.jpg   (242, 81)    219     87
3  00148.jpg   (251, 93)    247     98
```

This is same as case `(1)` and additionally we have the bounding box. We still need the mapping as discussed in `(1)`.

**Scenario #3 :** `start3` : csv with `2` or more feature points -

```
          FN         TL         TR          BL
0  00003.jpg   (108, 4)  (346, 13)  (102, 273)
1  00032.jpg   (81, 37)  (244, 27)   (73, 340)
2  00055.jpg  (190, 19)  (293, 21)  (148, 321)
3  00148.jpg  (228, 36)  (466, 31)  (220, 403)
```

We have three keypoints here and we could define a bounding box based on these. One common workflow could be to define it as the `min`-`max` of these keypoints.

### `scenarios` to `end`

Our task is to reach `end` starting from these scenarios and then finally arrive at `dirix`.

Let's import the package to get things started :

```python
import datatools
```

**From Scenario #1 :** To go from `start1` to `end`, we need to get to `start2` first. We would a parameter to setup bounding box sizes. There are three ways to setup those - Bounding box sizes depend on the image dimensions or depend on another header in the dataframe or fixed. For this, we can use  utility script `start1_to_start2.py` that also explores these variations.

**From Scenario #2 :** To go from `start2` to `end`,  we need to add information on the starting x-y coordinates of the bounding boxes. Basically, the bounding boxes would be in the surrounding of the feature point with some mapping that could be defined based on a paramater `ptformat`, like so :
```python
datatools.od.bboxcsv_to_minmaxcsv(indir, in_csv, out_csv, label = 'CP', ptformat = 'Center')
```

**From Scenario #3 :** To go from `start3` to `end` is pretty straight-forward, as we simply find the extents defined by them to set the bounding boxes :

```python
datatools.od.features_to_minmaxcsv(incsv, outcsv, feature_headers = ['TL', 'TR', 'BL'])

```

### `end` to `dirix`
Finally, to go from `end` to `dirix`,  run :

```python
datatools.od.minmaxcsv_to_setup(indir, in_csv, label, outdir_imgsxmls, outdir_debug)
```

### Schematic summary

Here's a schematic that summarizes listed workflows :

```
Scenario #1
     |
     v
Scenario #2 ----> end ----> dirix ----> tfrecords
                  ^
                  |  
Scenario #3 ------|
```

### Demos

Run `demos/objdetect_setup/bash rundemos.sh` to run the demos that showcase sample runs.

----

### Multiple objects

**Scenario #1 :** Input csv with multiple objects bounding boxes for each image.

Sample csv : 

```
          FN                 cat                  dog                  pig
0  00003.jpg  (108, 4, 200, 150)  (218, 23, 290, 120)  (158, 24, 260, 190)
1  00032.jpg  (38, 55, 160, 120)  (118, 53, 190, 220)  (128, 34, 192, 250)
```

Each bounding box is captured with this format - `[xmin, ymin, xmax, ymax]`.

Run `demos/demo_minmaxcsv_to_setup_multiple.py` for a sample case run.

Syntax :

```python
datatools.od.minmaxcsv_to_setup_multiple(
	indir,            # input directory of images
	in_csv,           # input csv of format shown above
	feature_labels,   # feature labels = ['cat', 'dog', 'pig'] here
	outdir_imgsxmls,  # output path to save images and xmls
	outdir_debug)     # output path to save bounding box debug images
```

Running it will generate `dirix` with xmls that have all feature labels' bounding boxes in each xml for an image.