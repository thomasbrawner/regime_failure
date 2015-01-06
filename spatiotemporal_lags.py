## ---------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------- ##
## file: spatial_lags.py
## purpose: construct spatiotemporal lags of authoritarian regime failure weighted by geographic distance
## author: thomas brawner
## date: 5 january 2015
## note: IPython Notebook available at: http://www.thomaswbrawner.com/spatiotemporal-lags.html
## ---------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------- ##

import requests, zipfile, StringIO
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print 'Python version ' + sys.version
print 'Pandas version ' + pd.__version__
print 'Numpy version ' + np.__version__

## ---------------------------------------------------------------------------------------------- ##
## download and load regimes data

gwf_data_url = 'http://sites.psu.edu/dictators/wp-content/uploads/sites/12570/2014/07/GWF-Autocratic-Regimes-1.2.zip'
r = requests.get(gwf_data_url)
z = zipfile.ZipFile(StringIO.StringIO(r.content))
z.extract('GWF Autocratic Regimes 1.2/GWFtscs.txt');
gwf = pd.read_table('GWF Autocratic Regimes 1.2/GWFtscs.txt')

## ---------------------------------------------------------------------------------------------- ##
## subset gwf to relevant columns

gwf = gwf[['cowcode','year','gwf_fail','gwf_enddate','gwf_fail_subsregime','gwf_fail_type']]

## ---------------------------------------------------------------------------------------------- ##
## apply specific date to observations

gwf['end_date'] = pd.to_datetime(gwf['gwf_enddate'], format = '%d%b%Y')
gwf['end_year'] = pd.DatetimeIndex(gwf['end_date']).year
gwf['date_year'] = gwf['end_year'] != gwf['year']
gwf['date'] = np.where(gwf['date_year'] == True, pd.to_datetime('12/31/' + gwf['year'].astype(str)), gwf['end_date'])
gwf.drop(['gwf_enddate','end_date','end_year','date_year'], axis = 1, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## generate transition variables of interest = dem/aut transition, coerce

gwf['coerce'] = ((gwf['gwf_fail_type'] == 4) | (gwf['gwf_fail_type'] == 5) | (gwf['gwf_fail_type'] == 6)).astype(int)
gwf['demtrans'] = (gwf['gwf_fail_subsregime'] == 1).astype(int)
gwf['auttrans'] = (gwf['gwf_fail_subsregime'] == 2).astype(int)
gwf.rename(columns = {'gwf_fail' : 'failure'}, inplace = True)
gwf.drop(['gwf_fail_type','gwf_fail_subsregime'], axis = 1, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## expand grid function

def expand_grid(x, y):
    xg, yg = np.meshgrid(x, y, copy = False)
    xg = xg.flatten()
    yg = yg.flatten() 
    return pd.DataFrame({'x' : xg, 'y' : yg})

## ---------------------------------------------------------------------------------------------- ##
## expand date

days = expand_grid(gwf['date'], gwf['date'])
days.columns = ['date1','date2']

## ---------------------------------------------------------------------------------------------- ##
## subset dates to observations within one calendar year

days = days[days['date2'] < days['date1']]
days = days[days['date2'] >= (pd.DatetimeIndex(days['date1']) - pd.tseries.offsets.DateOffset(years = 1))]

## ---------------------------------------------------------------------------------------------- ##
## expand cowcodes and merge

days = pd.merge(days, expand_grid(gwf['cowcode'], gwf['cowcode']), left_index = True, right_index = True)
days.rename(columns = {'x' : 'cowcode1', 'y' : 'cowcode2'}, inplace = True)
days = days[days['cowcode1'] != days['cowcode2']]

## ---------------------------------------------------------------------------------------------- ##
## expand failures indicator and merge, drop target state

days = pd.merge(days, expand_grid(gwf['failure'], gwf['failure']), left_index = True, right_index = True)
days.rename(columns = {'x' : 'failure1', 'y' : 'failure'}, inplace = True)
days.drop('failure1', axis = 1, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## expand coerced failure types and merge, drop target state

days = pd.merge(days, expand_grid(gwf['coerce'], gwf['coerce']), left_index = True, right_index = True)
days.rename(columns = {'x' : 'coerce1', 'y' : 'coerce'}, inplace = True)
days.drop('coerce1', axis = 1, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## expand democratic transitions, drop target state

days = pd.merge(days, expand_grid(gwf['demtrans'], gwf['demtrans']), left_index = True, right_index = True)
days.rename(columns = {'x' : 'demtrans1', 'y' : 'demtrans'}, inplace = True)
days.drop('demtrans1', axis = 1, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## expand authoritarian transitions, drop target state

days = pd.merge(days, expand_grid(gwf['auttrans'], gwf['auttrans']), left_index = True, right_index = True)
days.rename(columns = {'x' : 'auttrans1', 'y' : 'auttrans'}, inplace = True)
days.drop('auttrans1', axis = 1, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## expand years and merge

days = pd.merge(days, expand_grid(gwf['year'], gwf['year']), left_index = True, right_index = True)
days.rename(columns = {'x' : 'year1', 'y' : 'year2'}, inplace = True)
days.sort(['cowcode1','cowcode2','year1','year2'], inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## drop duplicate country-window observations for sending states

duplicate_check_cols = ['cowcode1','cowcode2','year1']
days.drop_duplicates(duplicate_check_cols, take_last = True, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## load and clean geographic distance data, generate unity-based normalized distance weight

dist = pd.read_csv('http://privatewww.essex.ac.uk/~ksg/data/capdist.csv')
dist = dist[['numa','numb','kmdist']]
dist['kmdist_norm'] = (dist['kmdist'] - np.min(dist['kmdist'])) / (np.max(dist['kmdist']) - np.min(dist['kmdist']))
dist['kmdist_norm'] = dist['kmdist_norm'] * -1 + np.max(dist['kmdist_norm'])
dist.drop('kmdist', axis = 1, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## construct the spatial lags for each fail variable == sum of weighted failures

days = pd.merge(days, dist, left_on = ['cowcode1','cowcode2'], right_on = ['numa','numb'], how = 'left')

out_list = []
for var in ['failure','coerce','demtrans','auttrans']:
    lags = days[['cowcode1','year1','kmdist_norm',var]]
    lags.loc[:,'weighted_lag'] = lags.loc[:,'kmdist_norm'] * lags.loc[:,var]
    lags = lags[['cowcode1','year1','weighted_lag']]
    lags = lags.groupby(['cowcode1','year1']).sum()
    lags.columns = ['lag_' + var]
    out_list.append(lags)

data = pd.concat(out_list, axis = 1)
data.reset_index(inplace = True)
data.rename(columns = {'cowcode1' : 'cowcode', 'year1' : 'year'}, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## plot distributions

plt.subplot(221)
plt.hist(data['lag_failure'], 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Failures')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(222)
plt.hist(data['lag_coerce'], 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Coerced Failures')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(223)
plt.hist(data['lag_demtrans'], 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Democratic Transitions')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(224)
plt.hist(data['lag_auttrans'], 40, normed = False, facecolor = 'k')
plt.xlabel('Lagged Autocratic Transitions')
plt.ylabel('Frequency')
plt.grid(True)

plt.show()

## ---------------------------------------------------------------------------------------------- ##
## write to file

data.to_csv('spatial_lags_failure.txt', sep = ',', index = False)

## ---------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------- ##
