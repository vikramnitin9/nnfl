import keras

from keras.applications.vgg16 import VGG16 as vgg
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Model

import os
from tqdm import tqdm

import _pickle as cPickle

base_model = vgg(include_top=True, weights='imagenet')

model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

files = []

for f in os.listdir('COCO/train2014'):

  files.append('COCO/train2014/' + f)

files.sort()

pbar = tqdm(total=len(files))
features = {}

for image_path in files:

  img = image.load_img(image_path, target_size=(224, 224))

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  features[image_path.split('/')[1]] = model.predict(x)

  pbar.update(1)

pbar.close()

with open('fc7-features.pkl', 'wb') as f:
  cPickle.dump(features, f, protocol=4)
