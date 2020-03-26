#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
# import numpy as np
# import tensorflow as tf

""" assuming its inside git repo (level 0)
    go one upper level (level 1), then
    and create imageFolder
"""

os.chdir("..")
print('path is ',os.getcwd())
if not os.path.exists("imageFolder"):
    os.makedirs("imageFolder")

os.chdir("imageFolder")

folders=['0','1','2','3','4','5']
for fol in folders:
    if not os.path.exists(fol):
        os.makedirs(fol)

os.chdir("..")

""" galaxy-zoo-the-galaxy-challenge folder is at level 1
"""

base='galaxy-zoo-the-galaxy-challenge/'
base1='imageFolder/'
# base2='/mnt/f/IITH/research/physics/galaxy_zoo/GalaxyClassification/imageFolder_medium/'

print("reading csv")
df = pd.read_csv(base+"training_solutions_rev1/training_solutions_rev1.csv")
print("read")

df.columns

# df=df[['GalaxyID', 'Class1.1', 'Class1.2']]

class0=df[
    (df['Class1.1']>0.469)
    & (df['Class7.1']>0.5)
]['GalaxyID'].values

class0.shape

class1=df[
    (df['Class1.1']>0.469)
    & (df['Class7.2']>0.5)
]['GalaxyID'].values

class1.shape

class2=df[
    (df['Class1.1']>0.469)
    & (df['Class7.3']>0.5)
]['GalaxyID'].values

class2.shape

class3=df[
    (df['Class1.2']>0.430)
    & (df['Class2.1']>0.602)
]['GalaxyID'].values

class3.shape

class4=df[
    (df['Class1.2']>0.430)
    & (df['Class2.2']>0.715)
    & (df['Class4.1']>0.619)
]['GalaxyID'].values

class4.shape

from shutil import copyfile

print("copying")

def cpy(file,folder):
    """
       file -> image file name
       folder -> folder name where to copy file
    """
    
    if type(file)!=str: file=str(file)
    if type(folder)!=str: folder=str(folder)
        
    copyfile(base+'images_training_rev1/'+file+'.jpg',
             base1+folder+'/'+file+'.jpg')

# converting filenames in dataFrame to string
class0=class0.astype(str)
class1=class1.astype(str)
class2=class2.astype(str)
class3=class3.astype(str)
class4=class4.astype(str)

print("done copying")

# def cpy_train_test(classdf,ratio=1,trainDes='trainset',testDes='testset'):
#     n=classdf.shape[0]
#     si=int(n*ratio) # split index
    
#     train=classdf[:si+1]
#     test=classdf[si+1:]
    
#     for f in train: cpy(f,trainDes)
#     for f in test: cpy(f,testDes)

# cpy_train_test(class0,ratio=0.9)
# cpy_train_test(class1,ratio=0.9)
# cpy_train_test(class2,ratio=0.9)
# cpy_train_test(class3,ratio=0.9)
# cpy_train_test(class4,ratio=0.9)

num_images=-1
for f in class0: cpy(f,0)
for f in class1: cpy(f,1)
for f in class2: cpy(f,2)
for f in class3: cpy(f,3)
for f in class4: cpy(f,4)

