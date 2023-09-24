import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



iris = load_iris() 
X = iris.data[:, (0, 1)] # petal length, petal width
y = ["malware", "non-malware"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
perceptron = Perceptron(0.001, 100)

perceptron.fit(X_train, y_train)

pred = perceptron.predict(X_test)


# X_train = []
# y_train = ["malware", "non-malware"]
# X_test = []
# y_test = []
# sk_perceptron = Perceptron()
# sk_perceptron.fit(X_train, y_train)
# sk_perceptron_pred = sk_perceptron.predict(X_test)

# Accuracy

accuracy_score(sk_perceptron_pred, y_test)