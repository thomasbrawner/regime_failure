import numpy as np
import pandas as pd
from itertools import product


class SpaceTimeLagger(object):

    def __init__(self, window, data):
        self.window = window 
        self.data = data

    def evaluate_event_window_gwf(self, dep_var, lag_var):
        events = self.data.query('{0} == 1'.format(lag_var))  # events of interest
        if dep_var == 'failure':
            expanded_dates = list(product(self.data['dateInt'], events['dateInt']))
        else:
            revised_date = np.where(self.data['demtrans'] == 0, pd.to_datetime('12/31/' + self.data['year'].astype(str)), self.data['date'])
            revised_date = pd.Series(revised_date)	
            self.data['dateInt'] = pd.to_datetime(revised_date).apply(lambda x: x.date().toordinal()) 
            expanded_dates = list(product(self.data['dateInt'], events['dateInt']))
        expanded_index = list(product(self.data.index.values, events.index.values))
        days = pd.concat([pd.DataFrame(expanded_index, columns=['idx_target', 'idx_sender']), 
                          pd.DataFrame(expanded_dates, columns=['date_target', 'date_sender'])], axis=1)

        days = pd.merge(days, events[['cowcode', 'failure', 'coerce', 'demtrans', 'auttrans']], left_on='idx_sender', right_index=True, how='left')
        days.rename(columns={'cowcode' : 'cowcode_sender'}, inplace=True)
        days = pd.merge(days, self.data[['cowcode', 'year']], left_on='idx_target', right_index=True, how='left')
        days.rename(columns={'cowcode' : 'cowcode_target'}, inplace=True)
        days = days.query('cowcode_target != cowcode_sender')  # drop for same state 
        days['time_lag'] = days.eval('date_target - date_sender')  # get window  
        days = days.query('time_lag > 0 & time_lag <= {0}'.format(str(self.window)))  # drop events outside window 
        duplicate_check_cols = ['cowcode_target', 'cowcode_sender', 'year']  # check for duplicates 
        days.drop_duplicates(duplicate_check_cols, take_last=True, inplace=True)  # drop duplicates 	
        self.dep_var = dep_var
        self.lag_var = lag_var
        self.days = days  

    def evaluate_event_window_coups(self, dep_var, lag_var):
        events = self.data.query('{0} == 1'.format(lag_var))  # events of interest
        if dep_var == 'attempt':
            expanded_dates = list(product(self.data['dateInt_coup'], events['dateInt_coup']))
        else:
            revised_date = np.where(self.data['success'] == 0, pd.to_datetime('12/31/' + self.data['year'].astype(str)), self.data['date_coup'])
            revised_date = pd.Series(revised_date)	
            self.data['dateInt_coup'] = pd.to_datetime(revised_date).apply(lambda x: x.date().toordinal()) 
            expanded_dates = list(product(self.data['dateInt_coup'], events['dateInt_coup']))
        expanded_index = list(product(self.data.index.values, events.index.values))
        days = pd.concat([pd.DataFrame(expanded_index, columns=['idx_target', 'idx_sender']), 
                          pd.DataFrame(expanded_dates, columns=['date_target', 'date_sender'])], axis=1)

        days = pd.merge(days, events[['cowcode', 'success', 'attempt']], left_on='idx_sender', right_index=True, how='left')
        days.rename(columns={'cowcode' : 'cowcode_sender'}, inplace=True)
        days = pd.merge(days, self.data[['cowcode', 'year']], left_on='idx_target', right_index=True, how='left')
        days.rename(columns={'cowcode' : 'cowcode_target'}, inplace=True)
        days = days.query('cowcode_target != cowcode_sender')  # drop for same state 
        days['time_lag'] = days.eval('date_target - date_sender')  # get window  
        days = days.query('time_lag > 0 & time_lag <= {0}'.format(str(self.window)))  # drop events outside window 
        duplicate_check_cols = ['cowcode_target', 'cowcode_sender', 'year']  # check for duplicates 
        days.drop_duplicates(duplicate_check_cols, take_last=True, inplace=True)  # drop duplicates 	
        self.dep_var = dep_var
        self.lag_var = lag_var
        self.days = days 

    def generate_lags(self, distance_weights, prefix): 
        if not hasattr(self, 'days'):
            raise Exception('evaluate_event_window method needs to be executed')
        if 'trade' in prefix:
            dist_lags = pd.merge(self.days, distance_weights, left_on=['cowcode_target', 'cowcode_sender', 'year'], right_on=['cowcode1', 'cowcode2', 'year'], how='left')	
        else:
            dist_lags = pd.merge(self.days, distance_weights, left_on=['cowcode_target', 'cowcode_sender'], right_on=['cowcode1','cowcode2'], how='left')
        dist_lags.rename(columns = {'cowcode_target' : 'cowcode', 'weight' : prefix + '_lag'}, inplace=True)
        dist_lags = dist_lags[['cowcode', 'year', (prefix + '_lag')]].groupby(['cowcode', 'year']).sum()
        return dist_lags.reset_index()

