from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from skip_thoughts import configuration
from skip_thoughts import encoder_manager

import numpy as np
from tqdm import tqdm
import _pickle as cPickle
import json

VOCAB_FILE = "skip_thoughts/skip_thoughts_bi/vocab.txt"
EMBEDDING_MATRIX_FILE = "skip_thoughts/skip_thoughts_bi/embeddings.npy"
CHECKPOINT_PATH = "skip_thoughts/skip_thoughts_bi/model.ckpt-500008"
# The following directory should contain files rt-polarity.neg and
# rt-polarity.pos.
MR_DATA_DIR = "/dir/containing/mr/data"

encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(bidirectional_encoder=True),
                   vocabulary_file=VOCAB_FILE,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                   checkpoint_path=CHECKPOINT_PATH)

from tqdm import tqdm

with open('COCO/annotations/captions_train2014.json') as json_data:
	data = json.load(json_data)

pbar = tqdm(total=len(data['annotations']))

# with shelve.open('stv-encodings-caption', protocol=4) as final_data:

final_data = {}

for dic in data['annotations']:

  iid = dic['image_id']
  caption = dic['caption']

  key = caption

  encoding = encoder.encode([caption])

  final_data[key] = encoding

  pbar.update(1)

pbar.close()

with open('stv-encodings.pkl', 'wb') as f:
  cPickle.dump(final_data, f)
