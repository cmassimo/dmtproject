from sklearn import preprocessing as pp
import numpy as np
import os
from dataset_filter import *

def feature_extraction(dset):

    dset = dset[dset['price_usd']<5000]
    field_list = ['promotion_flag', 'srch_length_of_stay', 'srch_booking_window', \
        'srch_adults_count', 'srch_children_count', 'prop_id', 'click_bool', 'booking_bool', \
        'prop_location_score2', 'srch_id']

    ds = dset[field_list]

    ds.loc[:, 'prop_review_score'] = dset['prop_review_score'].fillna(0)
    ds.loc[:, 'loc_ratio2'] = loc_ratio2(dset).fillna(0)
    ds.loc[:, 'norm_star_rating'] = norm_pcid(dset, 'prop_starrating')
    ds.loc[:, 'nlog_price'] = log_norm_srch_id(dset, 'price_usd').fillna(0)
    ds.loc[:, 'label'] = label(dset)

    return ds

def test_feature_extraction(dset):

    field_list = ['promotion_flag', 'srch_length_of_stay',
        'srch_booking_window', 'srch_adults_count',
        'srch_children_count', 'prop_id', \
        'prop_location_score2', 'srch_id']

    ds = dset[field_list]

    ds.loc[:, 'prop_review_score'] = dset['prop_review_score'].fillna(0)
    ds.loc[:, 'loc_ratio2'] = loc_ratio2(dset).fillna(0)
    ds.loc[:, 'norm_star_rating'] = norm_pcid(dset, 'prop_starrating')
    ds.loc[:, 'nlog_price'] = log_norm_srch_id(dset, 'price_usd').fillna(0)

    return ds

def scale_features(dset):
    field_list = ['prop_review_score', 'promotion_flag', 'srch_length_of_stay', \
        'srch_booking_window', 'srch_adults_count', 'srch_children_count', \
        'loc_ratio2']

    tmp = dset[field_list].astype(float).apply(pp.scale, axis=0, raw=True)

    tmp.loc[:, 'norm_star_rating'] = dset['norm_star_rating'].astype(float)
    tmp.loc[:, 'nlog_price'] = dset['nlog_price'].astype(float)
    tmp.loc[:, 'prop_id'] = dset['prop_id'].astype(float)
    tmp.loc[:, 'srch_id'] = dset['srch_id'].astype(float)
    tmp.loc[:, 'click_bool'] = dset['click_bool'].astype(float)
    tmp.loc[:, 'booking_bool'] = dset['booking_bool'].astype(float)
    tmp.loc[:, 'label'] = dset['label'].astype(float)

    return tmp

#watch out with wording: 'normalize' means scale rows to unit norm!!
def normalize_samples(dset):
    field_list = ['prop_review_score', 'promotion_flag', 'srch_length_of_stay', \
        'srch_booking_window', 'srch_adults_count', 'srch_children_count', \
        'loc_ratio2', 'norm_star_rating', 'nlog_price']

    tmp = dset[field_list].apply(lambda x: pp.normalize(x)[0], axis=1, raw=True)

    tmp.loc[:, 'prop_id'] = dset['prop_id']
    tmp.loc[:, 'srch_id'] = dset['srch_id']
    tmp.loc[:, 'click_bool'] = dset['click_bool']
    tmp.loc[:, 'booking_bool'] = dset['booking_bool']
    tmp.loc[:, 'label'] = dset['label']

    return tmp

def get_final_trainingset(dset, scale=True):
    if scale:
        fname = os.path.join('..', 'data', 'oversampled_scale.csv')
    else:
        fname = os.path.join('..', 'data', 'oversampled_noscale.csv')

    if os.path.isfile(fname):
        tset = pd.read_csv(fname)
        return tset
    else:
        if scale:
            tset = oversample_dataset(normalize_samples(scale_features(feature_extraction(dset))), fname)
        else:
            tset = oversample_dataset(feature_extraction(dset), fname)

        return tset

