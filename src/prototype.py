import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from data_preparation import *

whole_dataset = pd.read_csv(os.path.join('..', 'data', 'training_set_VU_DM_2014.csv'))
osn = get_final_trainingset(whole_dataset)

x = osn.filter(regex="[^(label)(booking_bool)(srch_id)(prop_id)(click_bool)]")
y = osn.filter(regex="label")

X, Y = shuffle(x, y, random_state=13)

offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], Y[:offset]
X_test, y_test = X[offset:], Y[offset:]

clf = ensemble.GradientBoostingClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
probs = clf.predict_proba(X_test)
mse = mean_squared_error(y_test, prediction)

