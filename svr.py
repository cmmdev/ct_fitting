import pprint, pickle
from sklearn import svm
import math

# read data
pkl_file = open('data.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

half_len = len(data)/2

# train
X = []
y = []
for [c0, c1, c2, c3, g0, g1, y0] in data[:half_len]:
    X.append([c0, c1])
    y.append(g0)

    X.append([c2, c3])
    y.append(g1)


# clf = svm.SVR(kernel='rbf', C=1e3, gamma=0.5)
clf = svm.SVR(kernel='linear')
clf.fit(X, y)

# validate
validate1 = []
validate2 = []
gt = []
for [c0, c1, c2, c3, g0, g1, y0] in data[half_len:]:
    validate1.append([c0, c1])
    validate2.append([c2, c3])
    gt.append(y0)

test1 = clf.predict(validate1)
test2 = clf.predict(validate2)

print ""
for i in range(len(test1)):
    print validate1[i], validate2[i], test1[i]/test2[i], gt[i], (gt[i]-test1[i]/test2[i])/gt[i] * 100


