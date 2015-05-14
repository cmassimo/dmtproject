from sklearn import preprocessing as pp
import os
from dataset_filter import *

dataset = pd.read_csv(os.path.join('..', 'data', 'training_set_VU_DM_2014.csv'))

def feature_extraction(dset):

    dset = dset[dset['price_usd']<5000]
    field_list = ['promotion_flag', 'srch_length_of_stay', 'srch_booking_window', \
        'srch_adults_count', 'srch_children_count', 'prop_id', 'booking_bool', 'prop_location_score2']

    ds = dset[field_list].astype(float64)

    ds.loc[:, 'prop_review_score'] = dset['prop_review_score'].fillna(0).astype(float64)
    ds.loc[:, 'loc_ratio2'] = loc_ratio2(dset).fillna(0).astype(float64)
    ds.loc[:, 'norm_star_rating'] = norm_pcid(dset, 'prop_starrating').astype(float64)
    ds.loc[:, 'nlog_price'] = log_norm_srch_id(dset, 'price_usd').fillna(0).astype(float64)
    ds.loc[:, 'label'] = label(dset).astype(float64)

    return ds

def scale_features(dset):
    field_list = ['prop_review_score', 'promotion_flag', 'srch_length_of_stay', \
        'srch_booking_window', 'srch_adults_count', 'srch_children_count', \
        'loc_ratio2']

    tmp = dset[field_list].apply(pp.scale, axis=0, raw=True)

    tmp.loc[:, 'norm_star_rating'] = dset['norm_star_rating']
    tmp.loc[:, 'nlog_price'] = dset['nlog_price']
    tmp.loc[:, 'prop_id'] = dset['prop_id']
    tmp.loc[:, 'booking_bool'] = dset['booking_bool']
    tmp.loc[:, 'label'] = dset['label']

    return tmp

def normalize_samples(dset):
    field_list = ['prop_review_score', 'promotion_flag', 'srch_length_of_stay', \
        'srch_booking_window', 'srch_adults_count', 'srch_children_count', \
        'loc_ratio2', 'norm_star_rating', 'nlog_price']

    tmp = dset[field_list].apply(lambda x: pp.normalize(x)[0], axis=1, raw=True)

    tmp.loc[:, 'prop_id'] = dset['prop_id']
    tmp.loc[:, 'booking_bool'] = dset['booking_bool']
    tmp.loc[:, 'label'] = dset['label']

    return tmp

def get_final_trainingset(dset):
    fname = os.path.join('..', 'data', 'oversampled_scalenorm.csv')

    if os.path.isfile(fname):
        tset = pd.read_csv(fname)
    else:
        tset = oversampled_dataset(normalize_samples(scale_features(feature_extraction(dset))), fname)

    return tset.drop('booking_bool', 1).drop('prop_id', 1)

