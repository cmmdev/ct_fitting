import pprint, pickle
from sklearn import svm

# read data
pkl_file = open('data.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

# train
X = []
y = []
for [c0, c1, c2, c3, y0] in data[:-100]:
    X.append([c0, c1, c2, c3])
    y.append(y0)

clf = svm.SVR(C=0.1, epsilon=0.01)
clf.fit(X, y)

# validate
validate = []
gt = []
for [c0, c1, c2, c3, y0] in data[-100:]:
    validate.append([c0, c1, c2, c3])
    gt.append(y0)

test = clf.predict(validate)

for i in range(len(test)):
    print validate[i], test[i], gt[i]


