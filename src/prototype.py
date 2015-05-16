import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_squared_error
from data_preparation import *
from result_calculation import *

whole_dataset = pd.read_csv(os.path.join('..', 'data', 'training_set_VU_DM_2014.csv'))
osn = get_final_trainingset(whole_dataset).filter(regex="[^(Unnamed: 0)]")

#dont get this -> why not srch_id and prop_id? We need this to calculate the order
#x = osn.filter(regex="[^(label)(booking_bool)(srch_id)(prop_id)(click_bool)]")
#shuy = osn.filter(regex="label")

#X, y = shuffle(osn.filter(regex="[^(label)]"), osn['label'], random_state=13)
X = osn.filter(regex="[^(label)]").values
y = osn['label'].values

folds = StratifiedKFold(y, 2, shuffle=True)

for train_index, test_index in folds:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #filter out unwanted columns (prop_id etc) -> because of ndarray has no label indexes
    X_train_clean = [x[:9] for x in X_train]
    X_test_clean = [x[:9] for x in X_test]

    clf = ensemble.GradientBoostingClassifier()
    clf.fit(X_train_clean, y_train)
    prediction = clf.predict(X_test_clean)
    probs = clf.predict_proba(X_test_clean)
    mse = mean_squared_error(y_test, prediction)


    #combining results: need to concatenate values and labels/indexes
    a = array([np.append(X_test[i], probs[i]) for i in range(X_test.shape[0])])
    keys = osn.keys().drop('label')
    keys = keys.append(array(['ignoring_prob', 'clicking_prob', 'booking_prob']))

    result = pd.DataFrame(data=a, columns=keys)

    #exporting
#    score = calculate_ndcg(order('booking', result))

