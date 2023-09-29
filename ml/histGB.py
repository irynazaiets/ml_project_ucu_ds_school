import json
import os

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

labels = ["malware", "non-malware"]
train = []
test = []
y = []
y_test = []

for file in os.listdir('ml/benigh')[:-2]:
    with open(f'ml/benigh/{file}') as file_with_features:
        train.append(json.load(file_with_features).get("API"))
        y.append(1)

for file in os.listdir('ml/benigh')[:-2]:
    with open(f'ml/benigh/{file}') as file_with_features:
            test.append(json.load(file_with_features).get("API"))
            y_test.append(1)

for file in os.listdir('ml/malware')[:-1]:
    with open(f'ml/malware/{file}') as file_with_features:
        train.append(json.load(file_with_features).get("API"))
        y.append(-1)

for file in os.listdir('ml/malware')[:-1]:
    with open(f'ml/malware/{file}') as file_with_features:
            test.append(json.load(file_with_features).get("API"))
            y_test.append(-1)

all_keys = set()

for d in train+test:
    all_keys.update(d.keys())

for d in train+test:
    for key in all_keys:
        if key not in d:
            d[key] = None

vec = DictVectorizer()
tr = vec.fit_transform(train).toarray()
tst = vec.fit_transform(test).toarray()

clf = HistGradientBoostingClassifier().fit(tr, y)
print(clf.score(tst, y_test))
