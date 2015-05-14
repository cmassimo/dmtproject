import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn import cross_validation
from data_preparation import *

#osn = pd.read_csv('../data/osn.csv')
#dataset = get_final_trainingset(osn)

dataset = pd.read_csv('../data/osn.csv')

x = dataset.filter(regex="[^(label)]")
y = dataset.filter(regex="label")
X, Y = shuffle(x, y, random_state=13)

offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], Y[:offset]
X_test, y_test = X[offset:], Y[offset:]

clf = ensemble.GradientBoostingClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
probs = clf.predict_proba(X_test)


