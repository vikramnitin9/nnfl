from threading import Thread
from cidereval import GuidanceCaption
import os
import sys
import _pickle as cPickle
from tqdm import tqdm

num_threads = 2
skip = 3

gc_dict = {}

with open('precomputed/fc7-features.pkl', 'rb') as f:
	features = cPickle.load(f)

fnames = list(features.keys())

print(sys.getsizeof(fnames), len(fnames), len(fnames[0]))

def compute(num, fnames):

	print("I am thread " + str(num))

	g = GuidanceCaption()

	pbar = tqdm(total=int(len(fnames)/skip))

	for i in range(num, len(fnames), skip):
		fname = fnames[i]
		tempname = g.getGC(fname, K=10)
		gc_dict[fname] = tempname
		# print(tempname)
		# print()
		pbar.update(1)

	pbar.close()

for num in range(0, num_threads):
	t = Thread(target=compute, args=(num,fnames))
	t.start()

with open('caption_dict.pkl', 'wb') as f:
	cPickle.dump(gc_dict, f)
