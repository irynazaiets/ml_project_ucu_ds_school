import json
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm

LIST_WITH_FEATURES = []
labels = ["malware", "non-malware"]

file_with_features = open("set_1.json")

data_from_file = json.load(file_with_features)
for i in data_from_file:
    LIST_WITH_FEATURES.append(i)

vec = DictVectorizer()
vec.fit_transform(data_from_file).toarray()
vector_with_feature_names = vec.get_feature_names_out()
file_with_features.close()

clf = svm.SVC()
clf.fit(vector_with_feature_names, labels)
print(clf)
# print(LIST_WITH_FEATURES)