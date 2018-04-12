# coding: utf-8

# In[1]:

import os
import numpy as np

# demo script for running CIDEr
import json
from pydataformat.loadData import LoadData
from pyciderevalcap.eval import CIDErEvalCap as ciderEval

pathToData = '../COCO/'
refName = 'annotations/captions_train2014.json'
df_mode = 'coco-val-df'
image_dir = 'train2014'

# In[2]:

# load reference and candidate sentences
# loadDat = LoadData(pathToData)
# ref = loadDat.readJson(refName)

ann_path = os.path.join(pathToData, refName)

data = json.load(open(ann_path, 'r'))

ann_data = data['annotations']

# print(ann_data[291]['caption'])

print(ann_data[0]['image'].keys())

image_ids = [ann['image']['file_name'] for ann in ann_data]
annotations = [ann['caption'] for ann in ann_data]

images = os.listdir(os.path.join(pathToData, image_dir))

with open('precomputed/features.pkl', 'rb') as f:
	features = cPickle.load(f)

for feature in features:


# candidates = ['Hello how are you', 'Today is a nice day']

# print(np.shape(ref))

# In[3]:

# calculate cider scores
scorer = ciderEval(gts, res, df_mode)
# # scores: dict of list with key = metric and value = score given to each
# # candidate
# scores = scorer.evaluate()

# print(scores['CIDEr'])

# In[7]:

# scores['CIDEr'] contains CIDEr scores in a list for each candidate
# scores['CIDErD'] contains CIDEr-D scores in a list for each candidate
