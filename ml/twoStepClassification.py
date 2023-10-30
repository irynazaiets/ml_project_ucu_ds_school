import json
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# def evaluate_metrics(clf, clfName, X_test, y_test):
#     y_pred = clf.predict(X_test)

#     print(f'Metrics for {clfName} classifier:')

#     print('Accuracy:', accuracy_score(y_test, y_pred))
#     print('F1 Score:', f1_score(y_test, y_pred,average='macro'))
#     print('Precision:', precision_score(y_test, y_pred,average='macro'))
#     print('Recall:', recall_score(y_test, y_pred,average='macro'))

#     return y_pred

def get_data_from_folder(folder_name, data_label, data_size_limit):
    global data, y
    for file in os.listdir(f'ml/{folder_name}'):
        with open(f'ml/{folder_name}/{file}') as file_with_features:
            data.append(json.load(file_with_features).get("API"))
            y.append(data_label)
            if list(os.listdir(f'ml/{folder_name}')).index(file) >= data_size_limit:
                break

benign_folder = 'benign'
malware_folder = 'malware'
puas_folder = 'PUAs'

file_process_limit = 1200 # use this variable to control, how many records to process from each file

data = []
y = []

get_data_from_folder(benign_folder, 1, file_process_limit)
get_data_from_folder(malware_folder, -1, file_process_limit)
get_data_from_folder(puas_folder, 0, file_process_limit)

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

#  divide by malware and non-malware
y_train1 = [1 if y == 0 else y for y in y_train]


# evaluate malware or not
histGradientBoostingClf = HistGradientBoostingClassifier().fit(X_train, y_train1)

y_pred1 = histGradientBoostingClf.predict(X_test)

X_train_nonmalware = [X_train[i] for i in range(len(X_train)) if y_train[i] != -1]
y_train_nonmalware = [y_train[i] for i in range(len(X_train)) if y_train[i] != -1]

histGradientBoostingClf2 = HistGradientBoostingClassifier().fit(X_train_nonmalware, y_train_nonmalware)

X_test2 = [X_test[i] for i in range(len(X_test)) if y_pred1[i] != -1]

y_pred2 = histGradientBoostingClf2.predict(X_test2)

y_final = y_pred1.copy()

j = 0
for i in range(len(y_final)):
    if y_final[i] != -1:
        y_final[i] = y_pred2[j]
        j += 1

print('Accuracy:', accuracy_score(y_test, y_final))
print('F1 Score:', f1_score(y_test, y_final,average='macro'))
print('Precision:', precision_score(y_test, y_final,average='macro'))
print('Recall:', recall_score(y_test, y_final,average='macro'))





# histGradientBoostingClf = HistGradientBoostingClassifier()
# randomForrestClf = RandomForestClassifier(max_depth=2, random_state=0)
# adaBoostClf = AdaBoostClassifier(random_state=0)
# mlpClf = MLPClassifier(random_state=1, max_iter=300)
# svmClf = svm.SVC()
# neighClf = KNeighborsClassifier(n_neighbors=3)
# logRegressionClf = LogisticRegression(random_state=0)

# classifiers = [histGradientBoostingClf, randomForrestClf, adaBoostClf, mlpClf, svmClf, neighClf, logRegressionClf]
# classifiers_names = ['HistGradientBoosting', 'RandomForest', 'AdaBoost', 'MLP', 'SVM', 'KNN', 'LogisticRegression']

# for clf, clfName in zip(classifiers, classifiers_names):
#     y_pred = cross_val_predict(clf, X, y, cv=10)

#     print(f'Cross validated metrics for {clfName} classifier:')

#     print('Accuracy:', accuracy_score(y, y_pred))
#     print('F1 Score:', f1_score(y, y_pred,average='macro'))
#     print('Precision:', precision_score(y, y_pred,average='macro'))
#     print('Recall:', recall_score(y, y_pred,average='macro'))
