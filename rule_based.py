from utils import *

TT = 0
FF = 0
for p in properties:
    T = 0
    F = 0
    for sel in train_data[['Entity_1', 'Entity_2']][train_data.Property==p].values:
        if predict_property(sel) == p:
            T += 1
        else:
            F += 1

    print ('%s, T=%d, F=%d, acc=%f' % (p, T, F, 1.0*T/(T+F)))
    TT += T
    FF += F
print ('T=%d, F=%d, acc=%f' % (TT, FF, 1.0*TT/(TT+FF)))


predictions = []
for sel in test_data[['Entity_1', 'Entity_2']].values:
    predictions.append(predict_property(sel))

sub = pd.DataFrame({'Id': test_data.Id, 'Property': predictions})
sub.to_csv('submit.csv', index=False)
