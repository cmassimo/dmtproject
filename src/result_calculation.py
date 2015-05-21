import pandas as pd
import numpy as np
import os

#example data
d = {'srch_id' : pd.Series([1, 1, 1, 1, 2, 2], range(6)), 'prop_id' : pd.Series([1, 53, 3, 8, 1, 8], index=range(6)), 
     'booking_prob' : pd.Series([0.042, 0.0001, 0.042, 0.0032, 0.12, 0.014], index=range(6)), 'clicking_prob' : pd.Series([0.234, 0.0023, 0.166, 0.078, 0.18, 0.054], index=range(6))}
df = pd.DataFrame(d)

#wrapper function
def order(order, dataset):
    '''order = 'booking' or 'score'
    '''
    if order == 'booking':
        return order_booking_only(dataset)
    if order == 'score':
        return order_score(dataset)
 
def order_booking_only(dataset):
    '''Calculates the ordering of hotels. Dataset should contain srch_id, prop_id,
    booking_prob, clicking_prop'''

    def sort_pos(x):
        tmp = x.sort(['booking_prob', 'clicking_prob'], axis=0, ascending=False)
        tmp['pos'] = range(1, len(x)+1)
        return tmp
    
    #sort by booking probability
    print "-- sorting properties --"
    df_sort = dataset.groupby('srch_id').apply(sort_pos)
    
    #file output
    df_sort[['srch_id','prop_id']].astype(int).to_csv(os.path.join('..', 'data', 'results.csv'), index=False)
    
    return df_sort
    
def order_score(dataset):
    '''Calculates the ordering of hotels. Dataset should contain srch_id, prop_id,
    booking_prob, clicking_prop'''
    
    #calculate score
    dataset['book_score'] = 5*dataset['booking_prob']*0.02791
    dataset['click_score'] = 1*dataset['clicking_prob']*0.01683
    dataset['score'] = dataset[['book_score', 'click_score']].max(axis=1)

    df_sort = dataset.groupby('srch_id').apply(lambda x: x.sort(['score'], axis=0, ascending=False))
    
    #file output
    df_sort[['srch_id','prop_id']].to_csv(os.path.join('..', 'data', 'results.csv'), index=False)
    
    return df_sort

def ndcg(group):
    score = (group['booking_bool']*5/group['pos']).sum()
    score += ((group['click_bool']-group['booking_bool'])/group['pos']).sum()

    click_sum = int((group['click_bool'].sum() - group['booking_bool'].sum()))

    if (group['booking_bool'].sum() > 0):
        opt = 5
    else:
        opt = 0

    for i in range(click_sum, 1, -1):
        opt += 1/float(i)

    if opt > 0:
        gain = float(score) / float(opt)
    else:
        gain = 0

    return gain

def calculate_ndcg(ordering):
    return ordering.groupby('srch_id').apply(ndcg).mean()

