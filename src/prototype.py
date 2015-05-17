import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_squared_error
import itertools
from data_preparation import *
from result_calculation import *

def expandgrid(names, *itrs):
    product = list(itertools.product(*itrs))
    return pd.DataFrame({names[i]: [x[i] for x in product] for i in range(len(names))}).values

pranges = {'learning_rate': [0.05, 0.1, 0.15], \
        'n_estimators': [200, 100, 50], \
        'max_depth': [4, 3, 2]}

grid = expandgrid(pranges.keys(), *pranges.values())

osn = get_final_trainingset(os.path.join('..', 'data', 'training_set_VU_DM_2014.csv'), False).filter(regex="[^(Unnamed: 0)]")

cols = ['promotion_flag', 'srch_length_of_stay', 'srch_booking_window',\
'srch_adults_count', 'srch_children_count', 'norm_star_rating',  \
 'prop_location_score2','prop_review_score','nlog_price',\
 'loc_ratio2', 'click_bool','prop_id', 'srch_id','booking_bool','label']

osn = osn[cols]

X = osn.filter(regex="[^(label)]").values
y = osn['label'].values

# with canopy you can skip the outer for
# initialize this variable and just run the inner code block,
# from line 33 onward
gparams = grid[0]

# params optimization
for gparams in grid:

    print "New 2FCV iteration, params:", gparams

    folds = StratifiedKFold(y, n_folds=2, shuffle=True)#, random_state=17)
    results = []

    # 2FCV
    for train_index, test_index in folds:

        print train_index
        print test_index
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #filter out unwanted columns (prop_id etc) -> because of ndarray has no label indexes
        X_train_clean = [x[:9] for x in X_train]
        X_test_clean = [x[:9] for x in X_test]

        clf = ensemble.GradientBoostingClassifier(learning_rate=float(gparams[0]), max_depth=int(gparams[1]), n_estimators=int(gparams[2]))
        clf.fit(X_train_clean, y_train)

        prediction = clf.predict(X_test_clean)
        probs = clf.predict_proba(X_test_clean)
        mse = mean_squared_error(y_test, prediction)

        print "predictions MSE:", mse

        #combining results: need to concatenate values and labels/indexes
        a = array([np.append(X_test[i], probs[i]) for i in range(X_test.shape[0])])
        keys = osn.keys().drop('label')
        keys = keys.append(array(['ignoring_prob', 'clicking_prob', 'booking_prob']))

        result = pd.DataFrame(data=a, columns=keys)

        # calculate the ordering
        print "Calculating the ordering based on the returned probabilities..."
        score = calculate_ndcg(order('booking', result))

        print "This model scored an NDCG of:", score

        results.append([score, params])

        print "-----"

# TODO integrate from alternative_prototype.py
# train final model picked from "results" on whole osn.
# test set load and final predictions
