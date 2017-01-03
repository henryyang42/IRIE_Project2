import pandas as pd
import numpy as np
import nltk
import jieba
import random
from jieba.analyse import set_stop_words
with open('data/ref_text.txt') as f:
    corpus = f.readlines()

def find_entity_in_corpus(sel, strict=False):
    sel = ' '.join(sel).replace('·', ' ').split(' ')
    for c in corpus:
        ct = 0
        for s in sel:
            if s in c:
                ct += 1
        if ct > 1:
            return c

    if not strict:
        for c in corpus:
            ct = 0
            for s in sel:
                if s in c:
                   return c

    return ''



train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
properties = ['birthPlace', 'spouse', 'parent', 'child', 'sibling', 'workPlace', 'deathPlace']

p_corpus = []
for i, p in enumerate(properties):
    temp_corpus = []
    selected = train_data[['Entity_1', 'Entity_2']][train_data.Property==p].values
    print (p, train_data.Property[train_data.Property==p].count())

    for sel in selected:
        match = find_entity_in_corpus(sel, True)
        if match:
            temp_corpus.append(match)
    print (len(temp_corpus))
    p_corpus.append(temp_corpus)

train_x = []
train_y = []
for i, p in enumerate(properties):
    for text in p_corpus[i]:
        train_x.append(text)
        train_y.append(i)


train = list(zip(train_x, train_y))

random.shuffle(train)

train_x, train_y = zip(*train)
train_x, train_y = list(train_x), list(train_y)

sub = pd.DataFrame({'train_x': train_x, 'train_y': train_y})
sub.to_csv('data/train_trim.csv', index=False)

def find_sample(p):
    print (properties[p])
    for i, v in enumerate(train_y):
        if p == v:
            return train_x[i]

birthPlace, spouse, parent, child, sibling, workPlace, deathPlace = range(7)

test_x = []
selected = test_data[['Entity_1', 'Entity_2']].values
for sel in selected:
    match = find_entity_in_corpus(sel, True)
    if not match:
        # Long sel[1] -> Place
        if len(sel[1]) > 5:
            match = find_sample(workPlace)
        # Same prefix -> family
        elif sel[0][0] == sel[1][0]:
            match = find_sample(np.random.choice([parent, child, sibling]))
        else:
            match = find_sample(spouse)
        print ('%s, %s' % (sel, match))
    test_x.append(match)

sub = pd.DataFrame({'Entity_1': test_data.Entity_1, 'Entity_2': test_data.Entity_2, 'Id': test_data.Id, 'test_x': test_x})
sub.to_csv('data/test_trim.csv', index=False)

