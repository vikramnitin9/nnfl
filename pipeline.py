from keras.applications.vgg16 import VGG16 as vgg
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.preprocessing import image
from keras.models import Model
import keras.backend as K
from keras.layers import Input, Dense, LSTM, Add, Multiply, Embedding, Concatenate, Lambda
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import _pickle as cPickle
import tqdm
import json
import re

depth = 512
regions = 10*10
stv_len = 2400

embedding_len = 512
vocab_size = 9568
lstm_hidden_neurons = embedding_len
max_len = 50

def attention_model(feature_maps, stv):
    # feature_maps ~ (depth, regions)
    # stv ~ (1, stv_len)

    fiwi = Dense(depth, use_bias=False, activation='linear')(feature_maps)  # (depth, depth)
    fsws = Dense(depth, use_bias=False, activation='linear')(stv)           # (1, depth)

    resultant = Add()([fiwi, fsws])  # BROADCAST words in Add() => (depth, depth)

    alpha = Dense(regions, use_bias=False, activation='softmax')(resultant)    # (depth, regions) # Softmax by default is applied on last axis

    z = Multiply()([alpha, feature_maps])                # Element-wise product => (depth, regions)
    z = Lambda(lambda z: K.sum(z, axis=-1))(z)           # Sum over all regions for each map => (depth,)
    z = Lambda(lambda z: K.reshape(z, (-1, 1, depth)))(z)

    return z

def decoder_model(context_vec, ground_truth):
    # context_vec ~ (depth,)

    # x(-1) = Wz * z
    x_minusone = Dense(embedding_len, use_bias=False, activation='linear')(context_vec) # (1, embedding_len)

    # xt = We * w(t) : Word embedding
    embeddings = Embedding(vocab_size, embedding_len)(ground_truth) # We need to define the input_shape here. (first layer)
    embeddings = Concatenate(axis=1)([x_minusone, embeddings])

    # h(t) = LSTM(xt, h(t-1)) : LSTM state
    lstm_out = LSTM(embedding_len, return_sequences=True)(embeddings)   # (embedding_len x embedding_len)

    # pt+1 = Softmax(Wh * h(t)) : LSTM predictions
    predictions = Dense(vocab_size, use_bias=False, activation='softmax')(lstm_out)

    return predictions


features_sym = Input(shape=(depth, regions), dtype='float32', name='feature_maps')
stv_sym = Input(shape=(1, stv_len), dtype='float32', name='stv')

context_vec = attention_model(features_sym, stv_sym)

ground_truth_sym = Input(shape=(None,), dtype='int32', name='ground_truth')

prediction = decoder_model(context_vec, ground_truth_sym)

model = Model(inputs=[features_sym, stv_sym, ground_truth_sym], outputs=prediction)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

vgg_model = vgg(include_top=False, weights='imagenet')

with open('precomputed/caption_dict.pkl', 'rb') as f:
	caption_dict = cPickle.load(f)

with open('precomputed/stv-encodings.pkl', 'rb') as f:
	stv_dict = cPickle.load(f)

ann_path = 'COCO/annotations/captions_train2014.json'
ann_data = json.load(open(ann_path, 'r'))['annotations']

vocab_path = 'skip_thoughts/skip_thoughts_bi/vocab.txt'
vocab = [line.rstrip('\n') for line in open(vocab_path)]

feature_maps_batch = np.array([])
stv_batch = np.array([])
ground_truth_batch = np.array([])

batch_size = 80
stv_skipped_count = 0

for ann in ann_data:
	image_id = ann['image_id']
	annotation = ann['caption']
	fname = "COCO_train2014_" + "{:012}".format(int(image_id)) + '.jpg'

	if fname not in caption_dict:
		continue

	img = image.load_img('./COCO/train2014/' + fname, target_size = (320, 320))

	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	feature_maps = vgg_model.predict(x)
	# print(feature_maps.shape)
	consensus_cap = caption_dict[fname][0]

	if consensus_cap not in stv_dict:
		stv_skipped_count += 1
		print("Warning : caption omitted because no STV available", stv_skipped_count)
		continue

	stv = stv_dict[consensus_cap][0]
	# print(len(stv))

	feature_maps = feature_maps.reshape((regions, depth))
	feature_maps = feature_maps.transpose() # (depth, regions)

	feature_maps_batch = np.append(feature_maps, feature_maps_batch)

	stv = stv.reshape((-1, stv_len))
	stv_batch = np.append(stv, stv_batch)

	words = annotation.split()

	words = map(lambda s: re.sub(r'[^\w\s]','',s), words)

	if any(word not in vocab for word in words):
		print("Warning : caption contains weird word")
		continue

	ground_truth = [vocab.index(word) for word in words]

	ground_truth_batch = np.append(ground_truth, ground_truth_batch)

	if (len(ground_truth_batch) == batch_size):
		ground_truth_batch = pad_sequences(ground_truth_batch, maxlen=50, padding='post', value=vocab.index('PAD'))
		ground_truth_batch = np.asarray(ground_truth_batch)
		ground_truth_ohe = [to_categorical(g) for g in ground_truth_batch]
		stv_batch = np.asarray(stv_batch)
		model.train_on_batch([feature_maps_batch, stv_batch, ground_truth_batch], ground_truth_ohe)
