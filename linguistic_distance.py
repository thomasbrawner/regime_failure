#!/usr/bin/python
# -*- coding: utf-8 -*-

## ---------------------------------------------------------------------- ##
## ---------------------------------------------------------------------- ##
## linguistic_distance.py
## [1] scrape statutory and de facto languages from ethnologue
## [2] binary connectivity for intersecting languages among dyads
## author: thomas brawner
## date: 5 december 2014
## input: country_codes.py
## output: country_languages.txt, lingustic_connectivity.txt
## ---------------------------------------------------------------------- ##
## ---------------------------------------------------------------------- ##

from bs4 import BeautifulSoup
import urllib2
import numpy as np
import pandas as pd
from country_codes import *

## ---------------------------------------------------------------------- ##
## tool to access website and read content

def access_website(url):
    try: 
        return urllib2.urlopen(url.encode('utf-8')).read()
    except:
        return ""

## ---------------------------------------------------------------------- ##
## remove special characters from country names & languages

def find_chars(text, chars):
    for key, value in chars.iteritems():
        text = text.replace(key, value)
    return text

character_dictionary = {'\xc3\xa0':'a',
                        '\xc3\xa2':'a',
                        '\xc3\xa3':'a',
                        '\xc3\x80':'A',
                        '\xc3\xa9':'e',
                        '\xc3\xa8':'e',
                        '\xc3\xaa':'e',
                        '\xc3\x89':'E',
                        '\xc3\xae':'i',
                        '\xc3\xad':'i',
                        '\xc3\xb4':'o',
                        '\xc3\xb2':'o',
                        '\xc3\xbb':'u',
                        '\xc3\xb9':'u',
                        '\xc5\x93':'oe',
                        '\xc3\xa7':'c',
                        '\xe2\x80\x93':'--',
                        '\xe2\x80\xa6':'...',
                        '\xe2\x80\x89?':'?',
                        '\xe2\x80\x89!':'!',
                        '\xe2\x80\x89:':':',
                        "\xe2\x80\x99":"'",
                        "\xc2\xab ":"``",
                        "\xc2\xab\xc2\xa0":"``",
                        "\xe2\x80\x9c":"``",
                        " \xc2\xbb":"''",
                        "\xc2\xa0\xc2\xbb":"''",
                        "\xe2\x80\x9d":"''",
                        "\xc2\xad":" ",
                        "\xc2\xa0":" ",}

## ---------------------------------------------------------------------- ##
## base link for all countries in the world

elink = 'http://www.ethnologue.com/browse/countries'

## ---------------------------------------------------------------------- ##
## collect links for each country on the site

links_list = []
countries_list = []
base_link = 'http://www.ethnologue.com'

soup = BeautifulSoup(access_website(elink))
countries = soup.find_all('div', attrs = {'class' : 'field-content'})

for country in countries:
    links_list.append(base_link + country.find('a').get('href'))
    country = country.find('a').text.encode('utf-8')
    country = find_chars(country, character_dictionary)
    countries_list.append(country)

## ---------------------------------------------------------------------- ##
## collect languages for each country

language_list = []

for link in links_list:
    souper = BeautifulSoup(access_website(link))
    languages = souper.find('div', attrs = {'class' : 'views-field views-field-field-national-languages'})
    languages = languages.find('div', attrs = {'class' : 'field-content'}).text.encode('utf-8')
    languages = find_chars(languages, character_dictionary)
    language_list.append(languages)

## ---------------------------------------------------------------------- ##
## convert to data frame

data = pd.DataFrame({'country' : pd.Series(countries_list),
                     'languages' : pd.Series(language_list)})

## ---------------------------------------------------------------------- ##
## generate cowcodes

data = generate_cowcode(data, 'country', dictionary_cowcodes)

## ---------------------------------------------------------------------- ##
## omit rows not captured, then omit country duplicates

data = data[(data['cowcode'].str.len() <= 3)]
data.drop_duplicates(inplace = True)

## ---------------------------------------------------------------------- ##
## sort on cowcode, reorder columns, write cleaned data to file

data['cowcode'] = data['cowcode'].astype(int)
data.sort(['cowcode'], inplace = True)
data = data[['cowcode','country','languages']]
data.to_csv('country_languages.txt', sep = ',', index = False)

## ---------------------------------------------------------------------- ##
## expand grid function 

def expand_grid(x, y):
    xg, yg = np.meshgrid(x, y, copy = False)
    xg = xg.flatten()
    yg = yg.flatten() 
    return pd.DataFrame({'x' : xg, 'y' : yg})

## ---------------------------------------------------------------------- ##
## expand cowcodes

dist = expand_grid(data['cowcode'], data['cowcode'])
dist.columns = ['cowcode1','cowcode2']
dist = dist[dist['cowcode1'] != dist['cowcode2']]

## ---------------------------------------------------------------------- ##
## expand languages

dist = pd.merge(dist, expand_grid(data['languages'], data['languages']), left_index = True, right_index = True)
dist.rename(columns = {'x' : 'language1', 'y' : 'language2'}, inplace = True)

## ---------------------------------------------------------------------- ##
## establish connectivity 

def language_connection(a, b):
    a = [i.strip() for i in a.split(',')]
    b = [i.strip() for i in b.split(',')]
    return len(set(a).intersection(b)) > 0

dist['connection'] = dist.apply(lambda x: language_connection(x['language1'], x['language2']), axis = 1)

## ---------------------------------------------------------------------- ##
## subset columns to dyad and distance metric, clean, and write to file

dist = dist[['cowcode1','cowcode2','connection']]
dist['connection'] = dist['connection'].astype(int)
dist.sort(['cowcode1','cowcode2'], inplace = True)

dist.to_csv('linguistic_connectivity.txt', sep = ',', index = False)

## ---------------------------------------------------------------------- ##
## ---------------------------------------------------------------------- ##
