from dataset_filter import *
from sklearn import preprocessing
import os

dataset = pd.read_csv(os.path.join('..', 'data', 'training_set_VU_DM_2014.csv'))

def feature_extraction(dset, scale=True):
    lean_ds = dset[dset['price_usd']<10000]

    fields = {
            'prop_review_score':   lean_ds['prop_review_score'],
            'promotion_flag':       lean_ds['promotion_flag'],
            'srch_length_of_stay':  lean_ds['srch_length_of_stay'],
            'srch_booking_window':  lean_ds['srch_booking_window'],
            'srch_adults_count':    lean_ds['srch_adults_count'],
            'srch_children_count':  lean_ds['srch_children_count'],
            'srch_room_count':      lean_ds['srch_room_count'],
            'loc_ratio2':           loc_ratio2(lean_ds)
            }

    ds = pd.DataFrame(fields)

    if scale:
        ds_scaled = preprocessing.scale(ds)
        ds_scaled['norm_star_rating']= norm_pcid(lean_ds, 'prop_star_rating')
        ds_scaled = preprocessing.normalize(ds_scaled)
        fname = 'oversampled_scaled.csv'
    else:
        ds_scaled = ds
        ds_scaled['norm_star_rating']= norm_pcid(lean_ds, 'prop_star_rating')
        fname = 'oversampled_nonscaled.csv'

    ds_scaled['nlog_price'] = log_norm_srch_id(lean_ds, 'price_usd')
    ds_scaled['booking_bool'] = lean_ds['booking_bool']
    ds_scaled['pro_id'] = lean_ds['prop_id']

    return oversampled_dataset(ds_scaled, fname)

