import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import re
import nltk
from nltk.corpus import stopwords
import string
import json
from time import time
import pickle
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.layers.merge import add

# Read Captions File

with open("descriptions.txt") as file:
    captions = file.read()

# creating a "descriptions" dictionary  where key is 'img_name' and value is list of captions corresponding to that image_file.

descriptions = {}
captions = captions.split('\n')
for ele in captions[:-1]:
    i_to_c = ele.split("\t")
    img_name = i_to_c[0].split(".")[0]
    cap = i_to_c[1]
    
    if descriptions.get(img_name) == None:
        descriptions[img_name] = []

    descriptions[img_name].append(cap)

# Data Cleaning

def clean_text(sample):
    sample = sample.lower()
    sample = re.sub("[^a-z]+"," ",sample)
    sample = sample.split()
    sample = [s for s in sample if len(s)>1]
    sample = " ".join(sample)
    return sample

#  modify all the captions i.e - cleaned captions

for key, desc_list in descriptions.items():
    for i in range(len(desc_list)):
        desc_list[i] = clean_text(desc_list[i])

#  writing clean description to .txt file

f = open("descriptions.txt","w")
f.write( str(descriptions) )
f.close()

#  reading description file

f = open("descriptions.txt", 'r')
descriptions = f.read()
f.close()

json_acceptable_string = descriptions.replace("'", "\"")
descriptions = json.loads(json_acceptable_string)

# finding the unique vocabulary 

vocabulary = set()

for key in descriptions.keys():
    [vocabulary.update(i.split()) for i in descriptions[key]]

print('Vocabulary Size: %d' % len(vocabulary))

#  ALl words in description dictionary
all_vocab =  []

for key in descriptions.keys():
    [all_vocab.append(i) for des in descriptions[key] for i in des.split()]

print('Vocabulary Size: %d' % len(all_vocab))
print(all_vocab[:15])

#  count the frequency of each word, sort them and discard the words having frequency lesser than threshold value

import collections

counter= collections.Counter(all_vocab)
dic_ = dict(counter)
threshelod_value = 10
sorted_dic = sorted(dic_.items(), reverse=True, key = lambda x: x[1])
sorted_dic = [x for x in sorted_dic if x[1]>threshelod_value]
all_vocab = [x[0] for x in sorted_dic]

print(len(all_vocab))

# Loading Training Testing Data

# TrainImagesFile
f = open(r"Flickr_8k.trainImages.txt")
train = f.read()
f.close()

# TestImagesFile
f = open(r"Flickr_8k.testImages.txt")
test = f.read()
f.close()

# splitting the train and test dataset

train = [e.split(".")[0] for e in train.split("\n")[:-1]]
test = [e.split(".")[0] for e in test.split("\n")[:-1]]

# Preparing description for Training Data

train_descriptions = {}

for t in train:
    train_descriptions[t] = []
    for cap in descriptions[t]:
        cap_to_append = "startseq " + cap + " endseq"
        train_descriptions[t].append(cap_to_append)

# Transfer Learning
# - Images -> Features
# - Text -> Features

# Step - 1 : Image Feature Extraction

model = ResNet50(weights="imagenet", input_shape=(224,224,3))
model.summary()

model_new = Model(inputs=model.input, outputs=model.layers[-2].output)

def preprocess_img(img):
    img = image.load_img(img, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def encode_image(img):
    img = preprocess_img(img)
    feature_vector = model_new.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    return feature_vector

#  running above function for all train images
#  this may take some time

start = time()
encoding_train = {}

for ix, img_id in enumerate(train):
    img_path = "Flicker8k_Dataset/" + img_id + ".jpg"
    encoding_train[img_id] = encode_image(img_path)

    if ix%100==0:
        print("Encoding image :" + str(ix))
        
end_t = time()
print("Total Time taken(sec) :", end_t-start)

#  running the same function for all test images

start = time()
encoding_test = {}

for ix, img_id in enumerate(test):
    img_path = "Flicker8k_Dataset/" + img_id + ".jpg"
    encoding_test[img_id] = encode_image(img_path)

    if ix%100==0:
        print("Test Encoding image :" + str(ix))
        
end_t = time()
print("Total Time taken(sec) :", end_t-start)

# save the bottleneck train features to disk

with open("encoded_train_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_train, encoded_pickle)

# save the bottleneck test features to disk

with open("encoded_test_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_test, encoded_pickle)

# Data pre-processing for Captions

word_to_idx = {}
idx_to_word = {}

for i, word in enumerate(all_vocab):
    word_to_idx[word] = i+1
    idx_to_word[i+1] = word

#  add two more word in vocabulary
    
word_to_idx['startseq'] = 1846
word_to_idx['endseq'] = 1847

idx_to_word[1846] = 'startseq'
idx_to_word[1847] = 'endseq'

# maximum length of a description in a dataset

max_len = 0
for key in train_descriptions.keys():
    for cap in train_descriptions[key]:
        max_len = max(max_len,len(cap.split()))

print(max_len)

# Load Glove Vectors

glove_dir = "glove.6B.50d.txt"
embedding_index = {} 

f = open(glove_dir, encoding='utf-8')

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float')
    embedding_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embedding_index))

# Get 50-dim dense vector for each of the 10000 words in out vocabulary

embedding_dim = 50
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in word_to_idx.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# define the RNN model

input_img_features = Input(shape=(2048,))
inp_img1 = Dropout(0.3)(input_img_features)
inp_img2 = Dense(256, activation='relu')(inp_img1)

# captions as input

input_captions = Input(shape=(max_len,))
inp_cap1 = Embedding(input_dim=vocab_size, output_dim=50, mask_zero=True)(input_captions)
inp_cap2 = Dropout(0.3)(inp_cap1)
inp_cap3 = LSTM(256)(inp_cap2)

decoder1 = add([inp_img2, inp_cap3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

# Combined model

model = Model(inputs=[input_img_features, input_captions], outputs=outputs)

# Important thing to note here, we are setting the weights of the embedding layer to the weights that we obtain from the GloVe model.
# Also, we will not train the weights of the pre-embedded layer in order to prevent overfitting.

model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer="adam")

#  data generator, intended to be used in a call to model.fit_generator()

def data_generator(train_descriptions, encoding_train, word_to_idx, max_len, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    while True:
        for key, desc_list in train_descriptions.items():
            n+=1
            photo = encoding_train[key]
            for desc in desc_list:
                seq = [word_to_idx[word] for word in desc.split() if word in word_to_idx]
                for i in range(1, len(seq)):
                    xi = seq[0:i]
                    yi = seq[i]
                    xi = pad_sequences([xi], maxlen=max_len, value=0, padding='post')[0]
                    yi = to_categorical([yi], num_classes=vocab_size)[0]
                    X1.append(photo)
                    X2.append(xi)
                    y.append(yi)
            if n==num_photos_per_batch:
                yield [[np.array(X1), np.array(X2)], np.array(y)]
                X1, X2, y = list(), list(), list()
                n=0

# Run the model

epochs = 10
num_photos_per_batch = 3
steps = len(train_descriptions)//num_photos_per_batch

# making a directory models to save our models
if not os.path.isdir("models"):
    os.mkdir("models")

for i in range(epochs):
    generator = data_generator(train_descriptions, encoding_train, word_to_idx, max_len, num_photos_per_batch)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save("models/model_" + str(i) + ".h5")

# functions to get the index and word for the predicted sequence
def predict_captions(image):
    start_word = ["startseq"]
    while True:
        par_caps = [word_to_idx[i] for i in start_word]
        par_caps = pad_sequences([par_caps], maxlen=max_len, padding='post')
        preds = model.predict([image, par_caps])
        word_pred = idx_to_word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "endseq" or len(start_word) > max_len:
            break
            
    return ' '.join(start_word[1:-1])

# function to plot the captions on the image
def plot_img_captions(img, model):
    img = encoding_test[img]
    img = img.reshape((1,2048))
    caption = predict_captions(img)
    i = Image.open("Flicker8k_Dataset/" + img + ".jpg")
    plt.imshow(i)
    plt.axis('off')
    plt.show()
    print(caption)

# load the model and call the above function
for i in range(10):
    try:
        img = test[i]
        plot_img_captions(img, model)
    except Exception as e:
        print(e)
        pass
