import pandas as pd
import numpy as np
import nltk
import jieba
import random
from jieba.analyse import set_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, Adamax
from keras.utils.np_utils import to_categorical

# Constants
n_svd = 500
batch_size = 32
nb_epoch = 300
class_ = 7
dropout = 0.4
valid_size = 350
properties = ['birthPlace', 'spouse', 'parent', 'child', 'sibling', 'workPlace', 'deathPlace']

jieba.analyse.set_stop_words('data/stopwords_zh.txt')
with open('data/stopwords_zh.txt') as f:
    stopwords = [s.strip() for s in f.readlines()]
train_data = pd.read_csv('data/train_trim.csv')
test_data = pd.read_csv('data/test_trim.csv')

train_x = list(train_data.train_x.values)
test_x = list(test_data.test_x.values)
train_y = list(train_data.train_y.values)

all_data = train_x + test_x
all_data = [' '.join([x for x in jieba.cut(d) if x not in stopwords]) for d in all_data]

# LSA
vectorizer = CountVectorizer()
all_data = vectorizer.fit_transform(all_data)
svd = TruncatedSVD(n_svd)
normalizer = Normalizer()
lsa = make_pipeline(svd, normalizer)
all_data = lsa.fit_transform(all_data)

train_x = all_data[:len(train_x)]
test_x = all_data[len(train_x):]

train_y = to_categorical(train_y, class_)
train_x, train_y, valid_x, valid_y = train_x[:-valid_size], train_y[:-valid_size], train_x[-valid_size:], train_y[-valid_size:]

model = Sequential()
model.add(Dense(output_dim=n_svd*2, input_dim=n_svd))
model.add(Activation("relu"))
model.add(Dropout(dropout))
model.add(Dense(output_dim=n_svd*4))
model.add(Activation("relu"))
model.add(Dropout(dropout))
model.add(Dense(output_dim=class_))
model.add(Activation("softmax"))

adam = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(train_x, train_y,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(valid_x, valid_y),
          shuffle=True)

predictions = model.predict(test_x)
predictions = np.argmax(predictions, axis=1)
predictions = [properties[p] for p in predictions]

sub = pd.DataFrame({'Entity_1': test_data.Entity_1, 'Entity_2': test_data.Entity_2, 'Id': test_data.Id, 'Property': predictions})
sub.to_csv('gg.csv', index=False)

sub = pd.DataFrame({'Id': test_data.Id, 'Property': predictions})
sub.to_csv('submit.csv', index=False)
