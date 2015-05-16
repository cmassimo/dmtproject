# -*- coding: utf-8 -*-
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from data_preparation import *
from result_calculation import *
from sklearn.grid_search import GridSearchCV

whole_dataset = pd.read_csv(os.path.join('..', 'data', 'training_set_VU_DM_2014.csv'))
test_data = pd.read_csv(os.path.join('..','data','test_set_VU_DM_2014.csv'))
osn = get_final_trainingset(whole_dataset, False).filter(regex="[^(Unnamed: 0)]")
test_set = test_feature_extraction(test_data)


#dont get this -> why not srch_id and prop_id? We need this to calculate the order
x = osn.filter(regex="[^(label)(booking_bool)(srch_id)(prop_id)(click_bool)]").values
y = osn.filter(regex="label")['label'].values

clf = ensemble.GradientBoostingClassifier()

tuned_parameters = {'learning_rate': [0.05, 0.1, 0.15], \
        'n_estimators': [200, 100, 50], \
        'max_depth': [4, 3, 2]}

grid_search = GridSearchCV(clf, param_grid=tuned_parameters, cv=2, scoring='mean_squared_error',n_jobs = 2)
grid_search.fit(x,y)


# load test set
# extract features from test set
# do final predictions

prediction = grid_search.predict(test_set)
probs = grid_search.predict_proba(test_set)

#combining results: need to concatenate values and labels/indexes
a = array([np.append(X_test[i], probs[i]) for i in range(X_test.shape[0])])
keys = osn.keys().drop('label')
keys = keys.append(array(['ignoring_prob', 'clicking_prob', 'booking_prob']))

result = pd.DataFrame(data=a, columns=keys)

#exporting
#ordering = order(result)
#total_averaged_score = ndcg(ordering)

