from time import time
import os, sys
import glob
import pandas as pd
import numpy as np

df_train  = pd.read_csv("train.csv",header=None,names=["file",'class']).set_index("file")

np.random.seed(100)
class_files = {}
for i in range(1,11,1):
    indexes = df_train[(df_train['class']==i)].index
    class_files[i] = np.random.choice(indexes,int(0.15*len(indexes)) ,replace=False)

import shutil 
directory = "val"
if not os.path.exists(directory):
    os.makedirs(directory)
    
    
for music_class in class_files:
    for i in class_files[music_class]:
        print (i)
        shutil.move("train/"+i, directory)

df = pd.DataFrame([[f,key] for key in class_files for f in class_files[key]],columns=['Id',"class"]).set_index("Id").sort_index()

df.to_csv("val.csv",header=None,index=True)
