import pandas as pd
import numpy as np
import jieba
import nltk
import random
from jieba.analyse import set_stop_words
jieba.analyse.set_stop_words('data/stopwords_zh.txt')
jieba.load_userdict('data/entities.txt')

with open('data/ref_text.txt') as f:
    corpus = f.readlines()

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
properties = ['birthPlace', 'spouse', 'parent', 'child', 'sibling', 'workPlace', 'deathPlace']
birthPlace, spouse, parent, child, sibling, workPlace, deathPlace = properties
PERSON, PLACE = 94, 87
entity_person = []
entity_place = []

workPlace_feature = ['学', '校', '院', '会', '司', '所', '寺', '室', '园', '馆', '机构', '中心', '天文台', '部', '组织', '集团', '厅', '实验', '局', '计画', '海军', '宫殿']

feature = np.load('data/feature.npy').item()

for p in [birthPlace, workPlace, deathPlace]:
    selected = train_data[['Entity_1', 'Entity_2']][train_data.Property==p].values
    entity_person.extend([sel[0] for sel in selected])
    entity_place.extend([sel[1] for sel in selected])

for p in [spouse, parent, child, sibling]:
    selected = train_data[['Entity_1', 'Entity_2']][train_data.Property==p].values
    entity_person.extend([sel[0] for sel in selected])
    entity_person.extend([sel[1] for sel in selected])

entity_person = np.unique(entity_person)
entity_place = np.unique(entity_place)

def get_entity_property(entity):
    if entity in entity_person:
        return PERSON
    if entity in entity_place:
        return PLACE
    for place_feature in workPlace_feature:
        if place_feature in entity:
            return PLACE
    return PERSON


def find_entity_in_corpus(entities, strict=True):
    entities = ' '.join(entities).replace('·', ' ').split(' ') # Split long entity like 巴伐利亚王后奥地利·埃斯特的玛丽亚·特蕾莎,奥地利的伊丽莎白·弗朗齐斯卡女大公 LOL
    for c in corpus:
        if all([e in c for e in entities]):
            return c
    if not strict:
        for c in corpus:
            if any([e in c for e in entities]):
                return c
    return ''

def predict_property(sel, topK=1000):
    props = (get_entity_property(sel[0]), get_entity_property(sel[1]))
    match = find_entity_in_corpus(sel)
    if props == (PERSON, PLACE):
        # hand craft workPlace feature
        for f in workPlace_feature:
            if f in sel[1]:
                return workPlace
        fct = [0]*2
        for i, p in enumerate([birthPlace, deathPlace]):
            for j, f in enumerate(feature[p][:topK]):
                if f in match:
                    fct[i] += (topK - j)
        return [birthPlace, deathPlace][np.argmax(fct)]
    else:
        family = [spouse, parent, child, sibling]
        if match:
            fct = [0]*4
            for i, p in enumerate(family):
                for j, f in enumerate(feature[p][:topK]):
                    if f in match:
                        fct[i] += (topK - j)
            return family[np.argmax(fct)]

        return np.random.choice(family)

print ('Utils loaded.')
