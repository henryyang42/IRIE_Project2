import pandas as pd
import numpy as np
import nltk
import jieba
from jieba.analyse import set_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD

jieba.load_userdict('data/entities.txt')
jieba.analyse.set_stop_words('data/stopwords_zh.txt')

with open('data/ref_text.txt') as f:
    corpus = f.readlines()
with open('data/entities.txt') as f:
    entities = [s.strip() for s in f.readlines()]

def find_entity_in_corpus(sel):
    sel = ' '.join(sel).replace('·', ' ').split(' ')
    match = ''
    for c in corpus:
        ct = 0
        for s in sel:
            if s in c:
                ct += 1
        if ct == len(sel):
            match += c
    if match:
        return match

    for c in corpus:
        ct = 0
        for s in sel:
            if s in c:
               match += c
               break

    return match


train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
all_data = train_data.append(test_data)
entities = np.concatenate((all_data.Entity_1.values, all_data.Entity_2.values))
properties = ['birthPlace', 'spouse', 'parent', 'child', 'sibling', 'workPlace', 'deathPlace']

p_corpus = []
for i, p in enumerate(properties):
    temp_corpus = []
    selected = train_data[['Entity_1', 'Entity_2']][train_data.Property==p].values
    print (p, train_data.Property[train_data.Property==p].count())

    for sel in selected:
        match = find_entity_in_corpus(sel)
        temp_corpus.append(match)
    print (len(temp_corpus))
    p_corpus.append(temp_corpus)

topK = 100
features = [jieba.analyse.extract_tags(' '.join(p), topK=topK) for p in p_corpus]

selected = test_data[['Entity_1', 'Entity_2']].values
prediction = []
randct = 0
for sel in selected:
    match = find_entity_in_corpus(sel)
    p = np.random.randint(0, 7)
    pct = [0]*7
    if match:
        print(match)
        s = match
        for pp in range(7):
            for keyword in features[pp]:
                if keyword in s:
                    pct[pp] += topK - features[pp].index(keyword)

    if np.sum(pct) != 0:
        p = np.argmax(pct)
    else:
        randct += 1
    print (pct, end=',')
    print (','.join([sel[0], sel[1], properties[p]]))
    prediction.append(properties[p])

sub = pd.DataFrame({'Id': test_data.Id, 'Property': prediction})
sub.to_csv('gg.csv', index=False)
