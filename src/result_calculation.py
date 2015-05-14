import pandas as pd
import numpy as np
import os

#example data
d = {'srch_id' : pd.Series([1, 1, 1, 1, 2, 2], range(6)), 'prop_id' : pd.Series([1, 53, 3, 8, 1, 8], index=range(6)), 
     'booking_prob' : pd.Series([0.042, 0.0001, 0.042, 0.0032, 0.12, 0.014], index=range(6)), 'clicking_prob' : pd.Series([0.234, 0.0023, 0.166, 0.078, 0.18, 0.054], index=range(6))}
df = pd.DataFrame(d)
 
def order_booking_only(dataset):
    '''Calculates the ordering of hotels. Dataset should contain srch_id, prop_id,
    booking_prob, clicking_prop'''
    
    #sort by booking probability    
    df_sort = dataset.groupby('srch_id').apply(lambda x: x.sort(['booking_prob', 'clicking_prob'], axis=0, ascending=False))
    
    #file output
    df_sort[['srch_id','prop_id']].to_csv(os.path.join('..', 'data', 'results.csv'), index=False)
