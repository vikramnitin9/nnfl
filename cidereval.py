import os
import numpy as np
import itertools

import _pickle as cPickle

from time import time

import json
from cider.pydataformat.loadData import LoadData
from cider.pyciderevalcap.eval import CIDErEvalCap as ciderEval
from cider.pyciderevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from cider.pyciderevalcap.cider.cider import Cider

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

		tokenizer = PTBTokenizer('gts')
		self.full_gts = tokenizer.tokenize(self.full_gts)

		pair_list = []

		for n in self.fnames:
			for c in captions[n]:
				pair_list.append((n,c))

		self.full_res = [{'image_id' : n, 'caption' : c} for n,c in pair_list]

		tokenizer = PTBTokenizer('res')
		self.full_res = tokenizer.tokenize(self.full_res)

	def getGC(self, fname, K=60):
		feature = self.features[fname]

		# t = time()
		dist = []
		for key2 in self.fnames:
			dist.append(np.sum((feature - self.features[key2]) ** 2))
		# print("dist", time()-t)

		ind = np.argpartition(dist, K)[:K]

		# t = time()
		nearest_names = self.fnames[ind]
		nearest_names = set(nearest_names)

		gts = {n : self.full_gts[n] for n in nearest_names}
		res = [r for r in self.full_res if r['image_id'] in nearest_names]
		# print("dics ", time()-t) #0.02

		# t = time()
		scorers = [
			(Cider(df=df_mode), "CIDEr")
		]
		# print("ciderobj", time()-t) # 0.00050

		# t = time()
		for scorer, method in scorers:
			scores = scorer.compute_score(gts, res)
		# print("scorerloop ", time()-t)  # 0.04

		# t = time()
		best_caption = res[np.argmax(scores)]['caption']
		# print("argmax", time()-t) # 0.0

		return best_caption
