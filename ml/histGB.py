import json
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

# todo: panda library
benign_folder = 'benign'
malware_folder = 'malware'

file_process_limit = 1200 # use this variable to control, how many records to process from each file

data = []
y = []

for file in os.listdir(f'ml/{benign_folder}'):
    with open(f'ml/{benign_folder}/{file}') as file_with_features:
        data.append(json.load(file_with_features).get("API"))
        y.append(1)
        if list(os.listdir(f'ml/{benign_folder}')).index(file) >= file_process_limit:
            break

for file in os.listdir(f'ml/{malware_folder}'):
    with open(f'ml/{malware_folder}/{file}') as file_with_features:
        data.append(json.load(file_with_features).get("API"))
        y.append(-1)
        if list(os.listdir(f'ml/{malware_folder}')).index(file) >= file_process_limit:
            break

all_keys = set()

for d in data:
    all_keys.update(d.keys())

for d in data:
    for key in all_keys:
        if key not in d:
            d[key] = None

vec = DictVectorizer()
X = vec.fit_transform(data).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = HistGradientBoostingClassifier().fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('AUC:', roc_auc_score(y_test, y_pred))

#cross validation
# 3 class