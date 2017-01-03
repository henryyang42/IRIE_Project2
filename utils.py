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

for p in ['birthPlace', 'workPlace', 'deathPlace']:
    selected = train_data[['Entity_1', 'Entity_2']][train_data.Property==p].values
    entity_person.extend([sel[0] for sel in selected])
    entity_place.extend([sel[1] for sel in selected])

for p in ['spouse', 'parent', 'child', 'sibling']:
    selected = train_data[['Entity_1', 'Entity_2']][train_data.Property==p].values
    entity_person.extend([sel[0] for sel in selected])
    entity_person.extend([sel[1] for sel in selected])

entity_person = np.unique(entity_person)
entity_place = np.unique(entity_place)

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

def get_property(s):
    if s in entity_person:
        return PERSON
    if s in entity_place:
        return PLACE
    for place_feature in workPlace_feature:
        if place_feature in s:
            return PLACE
    return PERSON

topK = 200000
w = 1
workPlace_feature = ['学', '校', '院', '会', '司', '所', '寺', '室', '园', '馆', '机构', '中心', '天文台', '部', '组织', '集团', '厅', '实验', '局', '计画', '海军', '宫殿']
deathPlace_feature = ['庇护', '老', '神话', '废黜' ,'亡', '世', '战', '年代', '去世', '史上', '死', '院', '逝', '击', '败', '终']*w \
+ jieba.analyse.extract_tags(
    ' '.join([find_entity_in_corpus(sel) for sel in train_data[['Entity_1', 'Entity_2']][train_data.Property==deathPlace].values]),
    topK=topK)

birthPlace_feature = []*w  \
+ jieba.analyse.extract_tags(
    ' '.join([find_entity_in_corpus(sel) for sel in train_data[['Entity_1', 'Entity_2']][train_data.Property==birthPlace].values]),
    topK=topK)


spouse_feature = ['公主', '妻子', '皇后', '婚', '夫人', '丈夫', '嫁', '育有' ,'两人', '夫妇', '离异', '姻', '驸马', '贵妃', '生']*w \
+ jieba.analyse.extract_tags(
    ' '.join([find_entity_in_corpus(sel) for sel in train_data[['Entity_1', 'Entity_2']][train_data.Property==spouse].values]),
    topK=topK)

parent_feature = ['亲', '母', '父', '生', '世', '世', '国王', '王后', '君主', '皇帝', '继承', '家族', '位', '祖']*w \
+ jieba.analyse.extract_tags(
    ' '.join([find_entity_in_corpus(sel) for sel in train_data[['Entity_1', 'Entity_2']][train_data.Property==parent].values]),
    topK=topK)

sibling_feature = ['弟', '哥', '兄', '同胞', '姊', '妹', '亲王', '异母', '次子']*w \
+ jieba.analyse.extract_tags(
    ' '.join([find_entity_in_corpus(sel) for sel in train_data[['Entity_1', 'Entity_2']][train_data.Property==sibling].values]),
    topK=topK)

child_feature = ['儿子', '生了', '之子', '女儿', '父亲', '母亲', '生母', '之女', '长女', '幼子', '养子', '次女', '产下']*w \
+ jieba.analyse.extract_tags(
    ' '.join([find_entity_in_corpus(sel) for sel in train_data[['Entity_1', 'Entity_2']][train_data.Property==child].values]),
    topK=topK)

def predict_property(sel):
    props = (get_property(sel[0]), get_property(sel[1]))
    match = find_entity_in_corpus(sel)
    #print (sel, end='')
    if props[1] == PLACE:
        #print ('(PERSON, PLACE)', end='')
        # workPlace
        for f in workPlace_feature:
            if f in sel[1]:
                return workPlace
        # birthPlace & deathPlace
        fct = [0]*2
        for i, feature in enumerate([birthPlace_feature, deathPlace_feature]):
            for j, f in enumerate(feature):
                if f in match:
                    fct[i] += (topK - j)
        return [birthPlace, deathPlace][np.argmax(fct)]

    else:
        #print ('(PERSON, PERSON)', end='')
        # No same prefix -> spouse
        #if sel[0][0] != sel[1][0]:
        #    return spouse
        #else: # Same prefix -> parent, child, sibling
        if match:
            fct = [0]*4
            for i, feature in enumerate([spouse_feature, parent_feature, child_feature, sibling_feature]):
                for j, f in enumerate(feature):
                    if f in match:
                        fct[i] += (topK - j)
            return [spouse, parent, child, sibling][np.argmax(fct)]

        if sel[0][0] != sel[1][0]:
            return spouse
        return np.random.choice([parent, child, child, sibling])

