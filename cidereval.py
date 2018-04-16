# coding: utf-8

# In[1]:

import os
import numpy as np
import itertools

import _pickle as cPickle

# demo script for running CIDEr
import json
from cider.pydataformat.loadData import LoadData
from cider.pyciderevalcap.eval import CIDErEvalCap as ciderEval

pathToData = 'COCO/'
refName = 'annotations/captions_train2014.json'
df_mode = 'coco-val-df'
image_dir = 'train2014'

class GuidanceCaption:

	def __init__(self):
		ann_path = os.path.join(pathToData, refName)

		ann_data = json.load(open(ann_path, 'r'))['annotations']

		image_ids = [ann['image_id'] for ann in ann_data]
		annotations = [ann['caption'] for ann in ann_data]
		names = ["COCO_train2014_" + "{:012}".format(int(i)) + '.jpg' for i in image_ids]

		captions = {}

		for i in range(len(annotations)):
			if names[i] in captions:
				captions[names[i]].append(annotations[i])
			else:
				captions[names[i]] = [annotations[i]]

		with open('precomputed/fc7-features.pkl', 'rb') as f:
			self.features = cPickle.load(f)

		self.fnames = np.array(list(self.features.keys()))

		self.full_gts = {n : [{'caption' : c} for c in captions[n]] for n in self.fnames}

		self.pair_list = []

		for n in self.fnames:
			for c in captions[n]:
				self.pair_list.append((n,c))

	def getGC(self, fname, K=60):
		print("Computing Guidance Caption for " + fname)

		if fname not in self.features:
			print("Error : unknown file " + fname)
			return

		dist = []
		for key2 in self.fnames:
			dist.append(np.sqrt(np.sum((self.features[fname] - self.features[key2]) ** 2)))

		ind = np.argpartition(dist, K)[:K]
		nearest_names = self.fnames[ind]

		gts = {n : self.full_gts[n] for n in nearest_names}

		res = [{'image_id' : n, 'caption' : c} for n,c in self.pair_list if n in nearest_names]

		scorer = ciderEval(gts, res, df_mode)
		scores = scorer.evaluate()['CIDEr']

		best_caption = res[np.argmax(scores)]['caption']
		return best_caption
