"""
    Generate train_trim.csv, test_trim.csv, feature.npy for later use.
"""
from utils import *

train_x = []
train_y = []
for i, p in enumerate(properties):
    found = 0
    selected = train_data[['Entity_1', 'Entity_2']][train_data.Property==p].values
    print (p, train_data.Property[train_data.Property==p].count())

    for entities in selected:
        match = find_entity_in_corpus(entities, True)
        if match and '足球' not in match: # why so many 足球 QAQ
            found += 1
            train_x.append(match)
            train_y.append(i)

    print (found)

sub = pd.DataFrame({'train_x': train_x, 'train_y': train_y})
sub.to_csv('data/train_trim.csv', index=False)

def find_sample(p):
    print (properties[p])
    for i, v in enumerate(train_y):
        if p == v:
            return train_x[i]

birthPlace, spouse, parent, child, sibling, workPlace, deathPlace = range(7)

not_match = 0
test_x = []
selected = test_data[['Entity_1', 'Entity_2']].values
for sel in selected:
    match = find_entity_in_corpus(sel, False)
    if not match:
        not_match += 1
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

print ('%d test data not matched.' % not_match)

sub = pd.DataFrame({'Entity_1': test_data.Entity_1, 'Entity_2': test_data.Entity_2, 'Id': test_data.Id, 'test_x': test_x})
sub.to_csv('data/test_trim.csv', index=False)


train_data = pd.read_csv('data/train_trim.csv')

feature = {}
for i, p in enumerate(properties):
    feature[p] = jieba.analyse.extract_tags(
    ' '.join([text for text in train_data.train_x[train_data.train_y==i].values]),
    topK=None)
    print ('%s %d' % (p, len(feature[p])))
np.save('data/feature.npy', feature)
'''
birthPlace 2977
spouse 2279
parent 2058
child 2440
sibling 1438
workPlace 1018
deathPlace 1596
'''

