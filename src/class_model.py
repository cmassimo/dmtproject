import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn import cross_validation
# import some data to play with
iris = datasets.load_iris()
X, Y = shuffle(iris.data, iris.target, random_state=13)

offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], Y[:offset]
X_test, y_test = X[offset:], Y[offset:]
'''
folds = cross_validation.StratifiedKFold(Y, n_folds=10)
for train_index, test_index in folds:
	X_train, X_test = X[train_index], X[test_index]
	Y_train, Y_test = Y[train_index], Y[test_index]
'''
clf = ensemble.GradientBoostingClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
probs = clf.predict_proba(X_test)
print prediction
print probs


