# -------------------------------------------------------------------------- ##

import numpy as np
import pandas as pd
from itertools import product

# -------------------------------------------------------------------------- ##

class SpaceTimeLagger(object):

	def __init__(self, window, gwf):
		self.window = window 
		self.gwf = gwf

	def evaluate_event_window(self, dep_var, lag_var):

		# locate the events of interest for making lags 
		events = self.gwf.query('{0} == 1'.format(lag_var))

		# combinations of observations and events 
		if dep_var == 'failure':
			expanded_dates = list(product(self.gwf['dateInt'], events['dateInt']))
		else:
			revised_date = np.where(self.gwf['demtrans'] == 0, pd.to_datetime('12/31/' + self.gwf['year'].astype(str)), self.gwf['date'])
			self.gwf['dateInt'] = revised_date.apply(lambda x: x.to_datetime().date().toordinal()) 
			expanded_dates = list(product(self.gwf['dateInt'], events['dateInt']))
		expanded_index = list(product(self.gwf.index.values, events.index.values))
		days = pd.concat([pd.DataFrame(expanded_index, columns=['idx_target', 'idx_sender']), 
                  		  pd.DataFrame(expanded_dates, columns=['date_target', 'date_sender'])], 
                  		  axis = 1)

		# merge events and regime information 
		days = pd.merge(days, events[['cowcode', 'failure', 'coerce', 'demtrans', 'auttrans']], 
						left_on='idx_sender', right_index=True, how='left')
		days.rename(columns={'cowcode' : 'cowcode_sender'}, inplace=True)
		days = pd.merge(days, self.gwf[['cowcode','year']], 
						left_on='idx_target', right_index=True, how='left')
		days.rename(columns={'cowcode' : 'cowcode_target'}, inplace=True)

		# drop for same state 
		days = days.query('cowcode_target != cowcode_sender')

		# drop events outside of window 
		days['time_lag'] = days.eval('date_target - date_sender')
		days = days.query('time_lag > 0 & time_lag <= {0}'.format(str(self.window)))

		# drop country-year duplicates 
		duplicate_check_cols = ['cowcode_target', 'cowcode_sender', 'year']
		days.drop_duplicates(duplicate_check_cols, take_last=True, inplace=True)
		
		self.dep_var = dep_var
		self.lag_var = lag_var
		self.days = days 

	def generate_lags(self, distance_weights, prefix): 

		if not hasattr(self, 'days'):
			raise Exception('evaluate_event_window method needs to be executed')

		if prefix == 'trade':
			dist_lags = pd.merge(self.days, distance_weights, left_on=['cowcode_target', 'cowcode_sender', 'year'], 
                     		 	 right_on=['cowcode1', 'cowcode2', 'year'], how='left')	
		else:
			dist_lags = pd.merge(self.days, distance_weights, left_on=['cowcode_target', 'cowcode_sender'], 
                     		 	 right_on=['cowcode1','cowcode2'], how='left')
		dist_lags.rename(columns = {'cowcode_target' : 'cowcode', 'weight' : prefix + '_lag'}, inplace=True)
		dist_lags = dist_lags[['cowcode', 'year', (prefix + '_lag')]].groupby(['cowcode', 'year']).sum()
		return dist_lags.reset_index()

# -------------------------------------------------------------------------- ##

