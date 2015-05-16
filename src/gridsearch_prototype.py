import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from data_preparation import *

whole_dataset = pd.read_csv(os.path.join('..', 'data', 'training_set_VU_DM_2014.csv'))

osn = get_final_trainingset(whole_dataset, False)

x = osn.filter(regex="[^(label)(prop_location_score2)(booking_bool)(srch_id)(prop_id)(click_bool)]")
y = osn.filter(regex="label")

clf = ensemble.GradientBoostingClassifier()

tuned_parameters = {'learning_rate': [0.05, 0.1, 0.15], \
        'n_estimators': [200, 100, 50], \
        'max_depth': [4, 3, 2]}

grid_search = GridSearchCV(clf, param_grid=tuned_parameters, cv=2, scoring='mean_squared_error')

grid_search.fit(x.values, y['label'].values)

grid_search


