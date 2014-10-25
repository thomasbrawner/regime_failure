## ---------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------- ##
## file: gwf_spatiotemporal_lags.py
## purpose: calculate the lagged rate of regime failure within defined geographic neighborhoods
## author: thomas brawner
## date: 25 october 2014
## note: IPython Notebook with details --> 'http://www.thomaswbrawner.com/spatiotemporal-lags.html'
## ---------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------- ##

import requests, zipfile, StringIO
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Math

print 'Python version ' + sys.version
print 'Pandas version ' + pd.__version__

## ---------------------------------------------------------------------------------------------- ##
## download and load regimes data

gwf_data_url = 'http://sites.psu.edu/dictators/wp-content/uploads/sites/12570/2014/07/GWF-Autocratic-Regimes-1.2.zip'
r = requests.get(gwf_data_url)
z = zipfile.ZipFile(StringIO.StringIO(r.content))
z.extract('GWF Autocratic Regimes 1.2/GWFtscs.txt');
gwf = pd.read_table('GWF Autocratic Regimes 1.2/GWFtscs.txt')

## ---------------------------------------------------------------------------------------------- ##
## download and load geographic distance data

dist = pd.read_csv('http://privatewww.essex.ac.uk/~ksg/data/capdist.csv')

## ---------------------------------------------------------------------------------------------- ##
## subset to relevant columns

gwf = gwf[['cowcode','year','gwf_fail','gwf_enddate']]
dist = dist[['numa','numb','kmdist']]

## ---------------------------------------------------------------------------------------------- ##
## apply specific date to observations

gwf['end_date'] = pd.to_datetime(gwf['gwf_enddate'], format = '%d%b%Y')
gwf['end_year'] = pd.DatetimeIndex(gwf['end_date']).year
gwf['date'] = pd.NaT
gwf['date_year'] = gwf['end_year'] != gwf['year']
gwf['date'][gwf['date_year'] == True] = pd.to_datetime('12/31/' + gwf['year'].astype(str))
gwf['date'][gwf['date_year'] == False] = gwf['end_date']
gwf.drop(['gwf_enddate','end_date','end_year','date_year'], axis = 1, inplace = True)
gwf.rename(columns = {'gwf_fail' : 'failure'}, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## historic windows

windows = np.arange(1, 11).tolist()

## ---------------------------------------------------------------------------------------------- ##
## expand grid function

def expand_grid(x, y):
    xg, yg = np.meshgrid(x, y, copy = False)
    xg = xg.flatten()
    yg = yg.flatten() 
    return pd.DataFrame({'x' : xg, 'y' : yg})

## ---------------------------------------------------------------------------------------------- ##
## expand date vector

days = expand_grid(gwf['date'], gwf['date'])
days.columns = ['date1','date2']
days = days[days['date2'] < days['date1']]
days = days[days['date2'] >= (pd.DatetimeIndex(days['date1']) - pd.tseries.offsets.DateOffset(years = windows[-1]))]

## ---------------------------------------------------------------------------------------------- ##
## expand cowcodes and merge

days = pd.merge(days, expand_grid(gwf['cowcode'], gwf['cowcode']), left_index = True, right_index = True)
days.rename(columns = {'x' : 'cowcode1', 'y' : 'cowcode2'}, inplace = True)
days = days[days['cowcode1'] != days['cowcode2']]

## ---------------------------------------------------------------------------------------------- ##
## expand failures indicator and merge

days = pd.merge(days, expand_grid(gwf['failure'], gwf['failure']), left_index = True, right_index = True)
days.rename(columns = {'x' : 'failure1', 'y' : 'failure2'}, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## expand years and merge

days = pd.merge(days, expand_grid(gwf['year'], gwf['year']), left_index = True, right_index = True)
days.rename(columns = {'x' : 'year1', 'y' : 'year2'}, inplace = True)
days.sort(['cowcode1','cowcode2','year1','year2'], inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## locate observations within a specific historical window

for year in windows:
    name = 'date_lag' + str(year)
    days[name] = pd.DatetimeIndex(days['date1']) - pd.tseries.offsets.DateOffset(years = year)
days['date_lag10'] = (days['date2'] < days['date_lag9'])
days['date_lag9'] = (days['date2'] < days['date_lag8']) & (days['date2'] >= days['date_lag9'])
days['date_lag8'] = (days['date2'] < days['date_lag7']) & (days['date2'] >= days['date_lag8'])
days['date_lag7'] = (days['date2'] < days['date_lag6']) & (days['date2'] >= days['date_lag7'])
days['date_lag6'] = (days['date2'] < days['date_lag5']) & (days['date2'] >= days['date_lag6'])
days['date_lag5'] = (days['date2'] < days['date_lag4']) & (days['date2'] >= days['date_lag5'])
days['date_lag4'] = (days['date2'] < days['date_lag3']) & (days['date2'] >= days['date_lag4'])
days['date_lag3'] = (days['date2'] < days['date_lag2']) & (days['date2'] >= days['date_lag3'])
days['date_lag2'] = (days['date2'] < days['date_lag1']) & (days['date2'] >= days['date_lag2'])
days['date_lag1'] = (days['date2'] >= days['date_lag1'])

## ---------------------------------------------------------------------------------------------- ##
## drop duplicate country-window observations for sending states

duplicate_check_cols = ['cowcode1','cowcode2','year1','date_lag1','date_lag2','date_lag3','date_lag4',
                        'date_lag5','date_lag6','date_lag7','date_lag8','date_lag9','date_lag10']
days.drop_duplicates(duplicate_check_cols, take_last = True, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## construct the lagged failure rates for each window & neighborhood

neighborhoods = (dist['kmdist'].describe(percentiles = [.10, .25, .50, .75])[4:8]).tolist()

out_list = []
for nn in neighborhoods:
    dist_new = dist[dist['kmdist'] <= nn]
    days_new = pd.merge(days, dist_new, left_on = ['cowcode1','cowcode2'], right_on = ['numa','numb'], how = 'left')
    days_new = days_new.dropna()
    for ww in windows:
        date_lag = 'date_lag' + str(ww)
        lags = days_new[days_new[date_lag] == True]
        lags = lags[['cowcode1','year1','failure2']]
        lags = lags.groupby(['cowcode1','year1']).agg('mean')
        lags.columns = ['lag_fail_n' + str(neighborhoods.index(nn)) + '_w' + str(ww)]
        out_list.append(lags)
        
data = pd.concat(out_list, axis = 1)
data.reset_index(inplace = True)
data.rename(columns = {'cowcode1' : 'cowcode', 'year1' : 'year'}, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## construct moving averages for specified windows using the lags above

ma_windows = [3, 5, 10]
hoods = np.arange(0, 4)

for nn in hoods:
    nn_names = 'lag_fail_n' + str(nn)
    subset_names = []
    for name in data.columns.tolist():
        if nn_names in name:
            subset_names.append(name)
    subset_data = data[subset_names]
    for ww in ma_windows:
        ww_names = []
        for name in subset_data.columns.tolist():
            if int(name.split('_w')[1]) <= ww:
                ww_names.append(name)
        subset_data2 = subset_data[ww_names]
        out_name = 'lag_ma_n' + str(nn) + '_w' + str(ww)
        data[out_name] = subset_data2.mean(axis = 1)

## ---------------------------------------------------------------------------------------------- ##
## clean and write to file

data.fillna(0, inplace = True)
data.to_csv('gwf_spatiotemporal_lags.txt')

## ---------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------- ##
