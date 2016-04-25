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
        'n_estimators': [200, 100], \
        'max_depth': [4, 2]}

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

results = []

# params optimization
for gparams in grid:

    print "New 2FCV iteration, params:", gparams

    folds = StratifiedKFold(y, n_folds=2, shuffle=True)#, random_state=17)
    score = 0

    # 2FCV
    for train_index, test_index in folds:

        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        
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
        a = array([np.append(a[i], 0) for i in range(X_test.shape[0])])
        keys = osn.keys().drop('label')
        keys = keys.append(array(['ignoring_prob', 'clicking_prob', 'booking_prob', 'pos']))

        result = pd.DataFrame(data=a, columns=keys)

        # calculate the ordering
        print "Calculating the ordering based on the returned probabilities..."
        score = calculate_ndcg(order('booking', result))

        print "This model scored an NDCG of:", score
        
#        score += (1.0 - mse)

    results.append([score/2.0, gparams])

    print "-----"

best_params = None
max_score = 0

for sc, par in results:
    print sc, par
    if sc > max_score:
        max_score = sc
        best_params = par

# for final prediction: 
best_params = [0.15, 4, 200]

print
print "Selecting best params tuple:", best_params
print "Score:", max_score
    
print "Fitting final model to whole dataset..."
# train final model on all training set with best params
X_train_clean = [x[:9] for x in X]
y_train = y
clf = ensemble.GradientBoostingClassifier(learning_rate=float(best_params[0]), max_depth=int(best_params[1]), n_estimators=int(best_params[2]))
clf.fit(X_train_clean, y_train)

# test ndcg on another slice of the training set

print "get NDCG from another slice of the training set..."
#fe = feature_extraction(os.path.join('..', 'data', 'training_set_VU_DM_2014.csv'))
#val_set = sample_dataset(fe, osn['prop_id'].unique())
val_set = pd.read_csv(os.path.join('..', 'data', 'validation.csv'))

cols = ['promotion_flag', 'srch_length_of_stay', 'srch_booking_window',\
'srch_adults_count', 'srch_children_count', 'norm_star_rating',  \
 'prop_location_score2','prop_review_score','nlog_price',\
 'loc_ratio2', 'click_bool','prop_id', 'srch_id','booking_bool','label']

val_set_sk = val_set[cols]
vset = val_set_sk.values

print "validation predictions..."
vset_clean = [x[:9] for x in vset]
val_prediction = clf.predict(vset_clean)
val_probs = clf.predict_proba(vset_clean)
val_mse = mean_squared_error(val_set['label'].values, val_prediction)

#combining results: need to concatenate values and labels/indexes
a = array([np.append(vset[i], val_probs[i]) for i in range(val_set.shape[0])])
a = array([np.append(a[i], 0) for i in range(val_set.shape[0])])
keys = val_set_sk.drop(val_set_sk['label']).keys()
keys = keys.append(array(['ignoring_prob', 'clicking_prob', 'booking_prob', 'pos']))

# calculate the ordering
print "Calculating the ordering based on the returned probabilities..."
val_result = pd.DataFrame(data=a, columns=keys)
val_score_1 = calculate_ndcg(order('booking', val_result))
print "NDCG for validation set (booking):", val_score_1

val_result = pd.DataFrame(data=a, columns=keys)
val_score_2 = calculate_ndcg(order('score', val_result))
print "NDCG for validation set (score):", val_score_2


# FINAL PREDICTIONS

print "Loading and extracting test data..."
# test set load and final predictions
test_data = pd.read_csv(os.path.join('..','data','test_set_VU_DM_2014.csv'))
test_set = test_feature_extraction(test_data)

cols = ['promotion_flag', 'srch_length_of_stay', 'srch_booking_window',\
'srch_adults_count', 'srch_children_count', 'norm_star_rating',  \
 'prop_location_score2','prop_review_score','nlog_price',\
 'loc_ratio2', 'prop_id', 'srch_id']

test_set = test_set[cols]
tset = test_set.values

print "Final predictions..."
tset_clean = [x[:9] for x in tset]
final_prediction = clf.predict(tset_clean)
final_probs = clf.predict_proba(tset_clean)

#combining results: need to concatenate values and labels/indexes
a = array([np.append(tset[i], final_probs[i]) for i in range(test_set.shape[0])])

keys = test_set.keys()
keys = keys.append(array(['ignoring_prob', 'clicking_prob', 'booking_prob']))

final_result = pd.DataFrame(data=a, columns=keys)

final_sorted = order('booking', final_result)

final_sorted[['srch_id','prop_id']].astype(int).to_csv(os.path.join('..', 'data', 'FINAL_SORTED.csv'), index=False)
print '------------------END'

#New 2FCV iteration, params: [  5.00000000e-02   4.00000000e+00   2.00000000e+02]
#predictions MSE: 0.639691730288
#predictions MSE: 0.63930613587
#-----
#New 2FCV iteration, params: [  5.00000000e-02   3.00000000e+00   2.00000000e+02]
#predictions MSE: 0.644046320652
#predictions MSE: 0.64231352201
#-----
#New 2FCV iteration, params: [  5.00000000e-02   2.00000000e+00   2.00000000e+02]
#predictions MSE: 0.652851735421
#predictions MSE: 0.65100687288
#-----
#New 2FCV iteration, params: [  1.00000000e-01   4.00000000e+00   2.00000000e+02]
#predictions MSE: 0.639611535254
#predictions MSE: 0.635857666429
#-----
#New 2FCV iteration, params: [  1.00000000e-01   3.00000000e+00   2.00000000e+02]
#predictions MSE: 0.639435106178
#predictions MSE: 0.638832973784
#-----
#New 2FCV iteration, params: [  1.00000000e-01   2.00000000e+00   2.00000000e+02]
#predictions MSE: 0.640742285238
#predictions MSE: 0.64578605054
#-----
#New 2FCV iteration, params: [  1.50000000e-01   4.00000000e+00   2.00000000e+02]
#predictions MSE: 0.637927439533
#predictions MSE: 0.637862590522
#-----
#New 2FCV iteration, params: [  1.50000000e-01   3.00000000e+00   2.00000000e+02]
#predictions MSE: 0.637783088471
#predictions MSE: 0.639546726761
#-----
#New 2FCV iteration, params: [  1.50000000e-01   2.00000000e+00   2.00000000e+02]
#predictions MSE: 0.640237056521
#predictions MSE: 0.641102547858
#-----
#New 2FCV iteration, params: [  5.00000000e-02   4.00000000e+00   1.00000000e+02]
#predictions MSE: 0.64740649259
#predictions MSE: 0.645545459649
#-----
#New 2FCV iteration, params: [  5.00000000e-02   3.00000000e+00   1.00000000e+02]
#predictions MSE: 0.655089176878
#predictions MSE: 0.652586753066
#-----
#New 2FCV iteration, params: [  5.00000000e-02   2.00000000e+00   1.00000000e+02]
#predictions MSE: 0.668826586258
#predictions MSE: 0.664279470379
#-----
#New 2FCV iteration, params: [   0.1    4.   100. ]
#predictions MSE: 0.640702187721
#predictions MSE: 0.638496146536
#-----
#New 2FCV iteration, params: [   0.1    3.   100. ]
#predictions MSE: 0.643428818888
#predictions MSE: 0.643067373469
#-----
#New 2FCV iteration, params: [   0.1    2.   100. ]
#predictions MSE: 0.653236671585
#predictions MSE: 0.649515209354
#-----
#New 2FCV iteration, params: [   0.15    4.    100.  ]
#predictions MSE: 0.63929877462
#predictions MSE: 0.636771911815
#-----
#New 2FCV iteration, params: [   0.15    3.    100.  ]
#predictions MSE: 0.642867453647
#predictions MSE: 0.638399910179
#-----
#New 2FCV iteration, params: [   0.15    2.    100.  ]
#predictions MSE: 0.644407198306
#predictions MSE: 0.64806364431
#-----
#New 2FCV iteration, params: [  0.05   4.    50.  ]
#predictions MSE: 0.667094373516
#predictions MSE: 0.662394841731
#-----
#New 2FCV iteration, params: [  0.05   3.    50.  ]
#predictions MSE: 0.674777057805
#predictions MSE: 0.672226989486
#-----
#New 2FCV iteration, params: [  0.05   2.    50.  ]
#predictions MSE: 0.689709373196
#predictions MSE: 0.691658713801
#-----
#New 2FCV iteration, params: [  0.1   4.   50. ]
#predictions MSE: 0.645096875601
#predictions MSE: 0.646387527768
#-----
#New 2FCV iteration, params: [  0.1   3.   50. ]
#predictions MSE: 0.654800474755
#predictions MSE: 0.652330122782
#-----
#New 2FCV iteration, params: [  0.1   2.   50. ]
#predictions MSE: 0.668377494066
#predictions MSE: 0.665490444532
#-----
#New 2FCV iteration, params: [  0.15   4.    50.  ]
#predictions MSE: 0.641993327773
#predictions MSE: 0.642305502314
#-----
#New 2FCV iteration, params: [  0.15   3.    50.  ]
#predictions MSE: 0.644094437672
#predictions MSE: 0.647831073116
#-----
#New 2FCV iteration, params: [  0.15   2.    50.  ]
#predictions MSE: 0.656749214089
#predictions MSE: 0.655890867972
#-----
