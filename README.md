## Regime failure

This repository will host several Python and R scripts for data management and analysis related to the breakdown of authoritarian regimes. Additions are made as they become available. Comments and questions are encouraged. 

29 June 2015

- `dissertation_programs.py`: Programs for estimating and presenting results for dissertation chapter 1 analysis.
- `spatiotemporal_lags.py`: Code to generate new lags data, updated to include all weights and principal components.
- Old scripts removed

16 December 2014

- `observed_case_simulation.R`: this script contains programs for simulated of predicted probabilities for logistic regression models and generalized additive models for binary dependent variables using the observed-case method. A short demonstration is available [here](http://www.thomaswbrawner.com/simulation.html 'Simulation demo').

5 December 2014

- `linguistic_distance.py`: this script scrapes the principal languages for all countries in the world from [Ethnologue](http://www.ethnologue.com/ 'Ethnologue: Languages of the World'), including statutory languages and de facto languages of national identity. In turn, it generates a measure of binary connectivity for all country dyads in the world, taking a value of one where there is an intersection in the principal languages of the respective countries and zero otherwise. This script relies on `country_codes.py`. 
- `country_codes.py`: this script includes a dictionary mapping country names to Correlates of War country codes as well as a function to generate the country codes for a provided *pandas* data frame. 

22 November 2014

- `kfold_cv_programs.R`: this is a collection of programs for conducting k-fold cross-validation, iterating k-fold cross-validation, and visualizing predictive performance. A short demo is available [here](http://www.thomaswbrawner.com/cross-validation.html 'k-fold CV demo').
 


