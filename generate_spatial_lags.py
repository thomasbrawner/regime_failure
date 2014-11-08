## ---------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------- ##
## file: generate_spatial_lags.py
## purpose: calculate spatial lags of regime failure, proportion of democracies in geographic neighborhoods
## author: thomas brawner
## date: 7 november 2014
## note: IPython Notebook with details --> 'http://www.thomaswbrawner.com/spatiotemporal-lags.html'
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
## download and load geographic distance data

dist = pd.read_csv('http://privatewww.essex.ac.uk/~ksg/data/capdist.csv')

## ---------------------------------------------------------------------------------------------- ##
## subset to relevant columns

gwf = gwf[['cowcode','year','gwf_fail','gwf_enddate','gwf_fail_subsregime','gwf_fail_type']]
dist = dist[['numa','numb','kmdist']]

## ---------------------------------------------------------------------------------------------- ##
## apply specific date to observations

gwf['end_date'] = pd.to_datetime(gwf['gwf_enddate'], format = '%d%b%Y')
gwf['end_year'] = pd.DatetimeIndex(gwf['end_date']).year
gwf['date'] = pd.NaT
gwf['date_year'] = gwf['end_year'] != gwf['year']
gwf.loc[gwf['date_year'] == True, 'date'] = pd.to_datetime('12/31/' + gwf['year'].astype(str))
gwf.loc[gwf['date_year'] == False, 'date'] = gwf['end_date']
gwf.drop(['gwf_enddate','end_date','end_year','date_year'], axis = 1, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## generate transition variables of interest = dem/aut transition, coerce

gwf['coerce'] = ((gwf['gwf_fail_type'] == 4) | (gwf['gwf_fail_type'] == 5) | (gwf['gwf_fail_type'] == 6)).astype(int)
gwf['demtrans'] = (gwf['gwf_fail_subsregime'] == 1).astype(int)
gwf['auttrans'] = (gwf['gwf_fail_subsregime'] == 2).astype(int)
gwf.rename(columns = {'gwf_fail' : 'failure'}, inplace = True)
gwf.drop(['gwf_fail_type','gwf_fail_subsregime'], axis = 1, inplace = True)

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

## ---------------------------------------------------------------------------------------------- ##
## subset dates to observations within maximum time window

days = days[days['date2'] < days['date1']]
days = days[days['date2'] >= (pd.DatetimeIndex(days['date1']) - pd.tseries.offsets.DateOffset(years = windows[-1]))]

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
## locate observations within a specific window

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
## construct the spatial lags for each fail variable, window, & neighborhood

neighborhoods = (dist['kmdist'].describe(percentiles = [.10, .25, .50, .75])[4:8]).tolist()

out_list = []
coerced_list = []
demtrans_list = []
auttrans_list = []

for nn in neighborhoods:
    dist_new = dist[dist['kmdist'] <= nn]
    days_new = pd.merge(days, dist_new, left_on = ['cowcode1','cowcode2'], right_on = ['numa','numb'], how = 'left')
    days_new = days_new.dropna()
    for ww in windows:
        date_lag = 'date_lag' + str(ww)
        lags = days_new[days_new[date_lag] == True]
        for var in ['failure','coerce','demtrans','auttrans']:
            lags2 = lags[['cowcode1','year1',var]]
            lags2 = lags2.groupby(['cowcode1','year1']).agg('mean')
            lags2.columns = ['stlag_' + var + '_n' + str(neighborhoods.index(nn)) + '_w' + str(ww)]
            out_list.append(lags2)
                
data = pd.concat(out_list, axis = 1)
data.reset_index(inplace = True)
data.rename(columns = {'cowcode1' : 'cowcode', 'year1' : 'year'}, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## construct moving averages for specified windows using the lags above

ma_windows = [3, 5, 10]
hoods = np.arange(0, 4)

for var in ['failure','coerce','demtrans','auttrans']:
    for nn in hoods:
        nn_names = 'stlag_' + var + '_n' + str(nn)
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
            out_name = 'stlag_ma_' + var + '_n' + str(nn) + '_w' + str(ww)
            data[out_name] = subset_data2.mean(axis = 1)

## ---------------------------------------------------------------------------------------------- ##
## merge the data frame of lags with the reduced gwf columns from above and clean up NAs

data = pd.merge(gwf, data, on = ['cowcode','year'], how = 'outer')
data.fillna(0, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------- ##
## now on to the proportion of democracies in the neighborhoods

z.extract('GWF Autocratic Regimes 1.2/GWF_AllPoliticalRegimes.txt');
gwf = pd.read_table('GWF Autocratic Regimes 1.2/GWF_AllPoliticalRegimes.txt')

## ---------------------------------------------------------------------------------------------- ##
## subset data frame columns

gwf = gwf[['cowcode','year','gwf_regimetype']]

## ---------------------------------------------------------------------------------------------- ##
## code democracies

gwf['democracy'] = np.nan
gwf.loc[pd.isnull(gwf['gwf_regimetype']), 'democracy'] = 1
gwf.loc[pd.notnull(gwf['gwf_regimetype']), 'democracy'] = 0
gwf.drop(['gwf_regimetype'], axis = 1, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## subset dictatorships

auth = gwf[gwf['democracy'] == 0]

## ---------------------------------------------------------------------------------------------- ##
## expand years, subset for same year

exdata = expand_grid(x = auth['year'], y = gwf['year'])
exdata.columns = ['year1','year2']
exdata = exdata[exdata['year1'] == exdata['year2']]

## ---------------------------------------------------------------------------------------------- ##
## expand cowcodes and merge on indices, omit dyads of same country

exdata = pd.merge(exdata, expand_grid(auth['cowcode'], gwf['cowcode']), left_index = True, right_index = True)
exdata.rename(columns = {'x' : 'cowcode1', 'y' : 'cowcode2'}, inplace = True)
exdata = exdata[exdata['cowcode1'] != exdata['cowcode2']]

## ---------------------------------------------------------------------------------------------- ##
## expand on democracy indicator and merge on indices

exdata = pd.merge(exdata, expand_grid(auth['democracy'], gwf['democracy']), left_index = True, right_index = True)
exdata.rename(columns = {'x' : 'democracy1', 'y' : 'democracy2'}, inplace = True)
exdata.sort(['cowcode1','year1','cowcode2'], inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## loop across neighborhoods, summarize democratic proportions by authoritarian country-year

out_list = []
for nn in neighborhoods:
    dist_new = dist[dist['kmdist'] <= nn]
    exdata_new = pd.merge(exdata, dist_new, left_on = ['cowcode1','cowcode2'], right_on = ['numa','numb'], how = 'left')
    exdata_new = exdata_new.dropna()
    exdata_new = exdata_new[['cowcode1','year1','democracy2']]
    dem_prop = exdata_new.groupby(['cowcode1','year1']).agg('mean')
    name_new = 'democ_prop_n' + str(neighborhoods.index(nn))
    dem_prop.rename(columns = {'democracy2' : name_new}, inplace = True)
    out_list.append(dem_prop)
    
## ---------------------------------------------------------------------------------------------- ##
## combine results & clean

democ = pd.concat(out_list, axis = 1)
democ.reset_index(inplace = True)
democ.rename(columns = {'cowcode1' : 'cowcode', 'year1' : 'year'}, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## merge with spatiotemporal lags of failures and write to file

data = pd.merge(data, democ, on = ['cowcode','year'], how = 'outer')
data.to_csv('gwf_spatial_data.txt')

## ---------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------- ##
