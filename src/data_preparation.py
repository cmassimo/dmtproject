from dataset_filter import *
from sklearn import preprocessing as pp
import os

dataset = pd.read_csv(os.path.join('..', 'data', 'training_set_VU_DM_2014.csv'))

def feature_extraction(dset):
    dset = dset[dset['price_usd']<5000]
    field_list = ['promotion_flag', 'srch_length_of_stay', 'srch_booking_window', \
        'srch_adults_count', 'srch_children_count', 'prop_id', 'click_bool', \
        'booking_bool']

    ds = dset[field_list].astype(float64)

    ds.loc[:, 'prop_review_score'] = dset['prop_review_score'].fillna(0).astype(float64)
    ds.loc[:, 'loc_ratio2'] = loc_ratio2(dset).fillna(0).astype(float64)
    ds.loc[:, 'norm_star_rating'] = norm_pcid(dset, 'prop_starrating').astype(float64)
    ds.loc[:, 'nlog_price'] = log_norm_srch_id(dset, 'price_usd').fillna(0).astype(float64)

    return ds

# still WIP
def scale_features(dset):
    field_list = ['prop_review_score', 'promotion_flag', 'srch_length_of_stay', \
        'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'prop_id', \
        'click_bool', 'booking_bool']

    return dset[field_list].apply(pp.scale, axis=0, raw=True)

# still WIP
def normalize_features(dset):
    return dset.apply(pp.normalize, axis=0, raw=True)

def get_final_trainingset(dset):
    return oversampled_dataset(normalize_features(scale_features(feature_extraction(dset))), '../data/oversampled_scalenorm.csv')

