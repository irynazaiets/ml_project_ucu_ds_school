import json
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

def predict(clf, clfName, X_test, y_test):
    y_pred = clf.predict(X_test)

    print(f'Metrics for {clfName} classifier:')

    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('F1 Score:', f1_score(y_test, y_pred,average='macro'))
    print('Precision:', precision_score(y_test, y_pred,average='macro'))
    print('Recall:', recall_score(y_test, y_pred,average='macro'))
    #print('AUC:', roc_auc_score(y_test, y_pred,multi_class='ovr',average='macro'))

benign_folder = 'benign'
malware_folder = 'malware'
puas_folder = 'malware'

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

for file in os.listdir(f'ml/{puas_folder}'):
    with open(f'ml/{puas_folder}/{file}') as file_with_features:
        data.append(json.load(file_with_features).get("API"))
        y.append(0)
        if list(os.listdir(f'ml/{puas_folder}')).index(file) >= file_process_limit:
            break

all_keys = set()

for d in data:
    all_keys.update(d.keys())

for d in data:
    for key in all_keys:
        if key not in d:
            d[key] = 0

vec = DictVectorizer()
X = vec.fit_transform(data).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

histGradientBoostingClf = HistGradientBoostingClassifier().fit(X_train, y_train)
randomForrestClf = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
adaBoostClf = AdaBoostClassifier(random_state=0).fit(X_train, y_train)

predict(histGradientBoostingClf, 'HistGradientBoostingClassifier', X_test, y_test)
predict(randomForrestClf, 'RandomForestClassifier', X_test, y_test)
predict(adaBoostClf, 'AdaBoostClassifier', X_test, y_test)


histGradientBoostingClf = HistGradientBoostingClassifier()
randomForrestClf = RandomForestClassifier(max_depth=2, random_state=0)
adaBoostClf = AdaBoostClassifier(random_state=0)

classifiers = [histGradientBoostingClf, randomForrestClf, adaBoostClf]
classifiers_names = ['HistGradientBoostingClassifier', 'RandomForestClassifier', 'AdaBoostClassifier']
scoring_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

for clf, clfName in zip(classifiers, classifiers_names):
    for metric in scoring_metrics:
        scores = cross_val_score(clf, X, y, cv=5, scoring=metric) # 10 fold, cross val predict
        print(f'Cross-validated {metric} for {clfName}: {scores.mean()}')
