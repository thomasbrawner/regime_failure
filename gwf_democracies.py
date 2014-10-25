## ---------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------- ##
## file: gwf_democracies.py
## purpose: calculate the proportion of democratic states within defined geographic neighborhoods
## author: thomas brawner
## date: 23 october 2014
## note: see IPython Notebook for details at `http://www.thomaswbrawner.com/democratic-neighborhoods.html`
## ---------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------- ##

import requests, zipfile, StringIO
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

print 'Python version ' + sys.version
print 'Pandas version ' + pd.__version__

## ---------------------------------------------------------------------------------------------- ##
## download and read gwf regimes data

gwf_data_url = 'http://sites.psu.edu/dictators/wp-content/uploads/sites/12570/2014/07/GWF-Autocratic-Regimes-1.2.zip'
r = requests.get(gwf_data_url)
z = zipfile.ZipFile(StringIO.StringIO(r.content))
z.extract('GWF Autocratic Regimes 1.2/GWF_AllPoliticalRegimes.txt')
gwf = pd.read_table('GWF Autocratic Regimes 1.2/GWF_AllPoliticalRegimes.txt')

## ---------------------------------------------------------------------------------------------- ##
## read distance data

dist = pd.read_csv('http://privatewww.essex.ac.uk/~ksg/data/capdist.csv')

## ---------------------------------------------------------------------------------------------- ##
## subset data frame columns

gwf = gwf[['cowcode','year','gwf_regimetype']]
dist = dist[['numa','numb','kmdist']]

## ---------------------------------------------------------------------------------------------- ##
## code democracies

gwf['democracy'] = np.nan
gwf['democracy'][pd.isnull(gwf['gwf_regimetype'])] = 1
gwf['democracy'][pd.notnull(gwf['gwf_regimetype'])] = 0
gwf.drop(['gwf_regimetype'], axis = 1, inplace = True)

## ---------------------------------------------------------------------------------------------- ##
## subset dictatorships

auth = gwf[gwf['democracy'] == 0]

## ---------------------------------------------------------------------------------------------- ##
## expand grid function

def expand_grid(x, y):
    xg, yg = np.meshgrid(x, y, copy = False)
    xg = xg.flatten()
    yg = yg.flatten() 
    return pd.DataFrame({'x' : xg, 'y' : yg})

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
## set up neighborhoods

neighborhoods = (dist['kmdist'].describe(percentiles = [.10, .25, .50, .75])[4:8]).tolist()

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
## combine results, clean up, write to file

data = pd.concat(out_list, axis = 1)
data.reset_index(inplace = True)
data.rename(columns = {'cowcode1' : 'cowcode', 'year1' : 'year'}, inplace = True)

data.to_csv('gwf_democratic_neighborhoods.txt')

## ---------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------- ##
