import pprint, pickle
from sklearn import svm
import math


# read data
pkl_file = open('data.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

half_len = len(data) / 2

# train
X = []
y = []
for [c0, c1, y0] in data[:half_len]:
    X.append([c0, c1])
    y.append(y0)

clf = svm.SVR(kernel='linear')
clf.fit(X, y)

# validate
validate = []
gt = []
for [c0, c1, y0] in data[half_len:]:
    validate.append([c0, c1])
    gt.append(y0)

test = clf.predict(validate)

for i in range(len(test)):
    print validate[i], test[i], gt[i], abs(test[i] - gt[i])

print len(data)



# plot


