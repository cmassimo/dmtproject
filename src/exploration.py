import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pylab
from math import log

from dataset_filter import sample_dataset

#load the entire dataset
#ds = pd.read_csv('training_set_VU_DM_2014.csv')

#load a smaller dataset
ds = sample_dataset(os.path.join('..', 'data', 'training_set_VU_DM_2014.csv'), os.path.join('..', 'data' , 'unbalanced_small_dataset.csv'))

def STATAclean(ds):
    STATAds = ds
    for num in range(1,9):
        STATAds = STATAds.drop('comp'+str(num)+'_rate', 1).drop('comp'+str(num)+'_inv', 1).drop('comp'+str(num)+'_rate_percent_diff', 1)
    
    STATAds = STATAds.replace(to_replace='NaN', value='.')
    
    STATAds.to_csv(os.path.join('..', 'data', 'STATA_small_dataset.csv'), sep=',')

    print "New STATA file created"

#############################

#first 5 entries
ds.head()

#to count missing values:
#ds['comp1_rate_percent_diff'].isnull().sum()

#filter stuff
randomfilter = ds[ds['random_bool'] == 1]
orderedfilter = ds[ds['random_bool'] == 0]

#difference of likelyhood of booking of already ranked queries and random ordered queries
orderedfilter['booking_bool'].mean() - randomfilter['booking_bool'].mean()

#correlations
ds[['booking_bool', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_length_of_stay', 'srch_saturday_night_bool', 'orig_destination_distance']].corr()

#############
#plotting

#count proportion of bookings -> need to count the occurences of a particular value before plotting
ds['booking_bool'].value_counts()

#booking likelyhood
ds['booking_bool'].value_counts().plot(kind='pie', figsize=(5,5), labels=['not booked', 'booked'])

#length of stay
ds[ds['srch_length_of_stay'] < 20]['srch_length_of_stay'].value_counts().plot(kind='pie', figsize=(5,5))

#review scores
ds['prop_review_score'].value_counts()
ds['prop_review_score'].value_counts().plot(kind='pie', figsize=(5,5))

#star ratings
ds['prop_starrating'].value_counts().plot(kind='pie', figsize=(5,5))

#plot of historical price
ds['prop_log_historical_price'].describe()
#not sold hotels are counted as 0
ds[ds['prop_log_historical_price'] != 0]['prop_log_historical_price'].plot(kind='box')
#save figure with no-whitespace around picture
pylab.savefig(os.path.join('..', 'figures', 'price_boxplot.png'), bbox_inches='tight')

#correlations
ds[ds['booking_bool'] == 1][['visitor_hist_adr_usd', 'prop_log_historical_price']].corr()

def log_norm_srch_id(dataset, key):
    ''' dataset = pandas datafram
        key = index of variable to be normalized (string)
        Returns normalized log of column by srch_id.
    '''
    nlog = dataset[key].apply(lambda x: log(x+1))
    #normalize by each srch_id
    nlog = dataset[key].groupby(dataset['srch_id']).apply(lambda x: (x-x.mean())/x.std())
    
    return nlog

ds['nlog_price'] = log_norm_srch_id(ds, 'price_usd')
ds['nlog_price'].plot(kind='hist', bins=25)

#attempt to engineer some composite of location and price
#center the price between 1 and 2
ds['nlog_price_center'] = ds['nlog_price'].groupby(ds['srch_id']).apply(lambda x: (x - min(x))/(max(x) - min(x))+1)
ds['nlog_price_center'].describe()

ds['loc_ratio1'] = ds['prop_location_score1'] / ds['nlog_price_center']
ds['loc_ratio2'] = ds['prop_location_score2'] / ds['nlog_price_center']

#with quality metrics
ds['loc_ratio2_rating'] = ds['loc_ratio2'] * ((ds['prop_review_score'] + ds['prop_starrating']) / 2)
#best correlation with booking: 0.078




    
