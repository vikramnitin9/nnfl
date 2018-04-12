from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from skip_thoughts import configuration
from skip_thoughts import encoder_manager

import numpy as np

with open('features.pkl', 'rb') as f:
	features = cPickle.load(f)



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

data = ["Hello how are you", "It is a good day"]
encodings = encoder.encode(data)
