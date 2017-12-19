from sklearn import svm

import random
import pprint


X = [[random.random(), random.random()] for i in range(200)]
y = [a/b for [a, b] in X]
clf = svm.SVR(kernel='rbf')
clf.fit(X,y)

pprint.pprint(X)
pprint.pprint(y)

print clf.predict([[0.7, 0.15]])

