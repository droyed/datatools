import pandas as pd
import numpy as np
import sys

in_csv = 'data_test/data_CP.csv' # csv path of feature points

## Sample input csv :
## FN : Filename
## CP : Feature    
#
#           FN          CP
# 0  00003.jpg   (165, 76)
# 1  00032.jpg   (128, 67)
# 2  00055.jpg   (242, 81)
# 3  00148.jpg   (251, 93)
# 4  00175.jpg  (304, 201)

# Read in csv as dataframe
df_in = pd.read_csv(in_csv)

optionID = 2
if len(sys.argv)>1:
    optionID = int(sys.argv[1])

#-------------------------------- Setup bboxH and bboxW headers ---------------
## Choose one among these 3 choices, edit and comment out other 2.
## Listed ones are just sample cases.
if optionID == 1:
    # Option #1 : As ratios of some header
    df_in['dist'] = np.random.randint(400,500,len(df_in))
    df_in['bboxH'] = 0.5*df_in['dist']
    df_in['bboxW'] = 0.2*df_in['dist']
elif optionID == 2:
    # Option #2 : As constant values
    df_in['bboxH'] = 200
    df_in['bboxW'] = 100

elif optionID == 3:
    # Option #3 : As ratios of image dimensions
    df_in['bboxH'] = 0.4 # 0.4 of image height
    df_in['bboxW'] = 0.3 # 0.4 of image width
else:
    print('No change to input df!')
    pass

print('Processed dataframe head :')
print(df_in.head())

out_csv = in_csv.replace('.csv','_bboxHW.csv')
df_in.to_csv(out_csv, index=False)
print('Output csv saved at - '+out_csv)
