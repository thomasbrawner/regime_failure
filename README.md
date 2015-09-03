## Regime failure

This repository will host several Python and R scripts for feature engineering, analysis, and visualization related to the analysis of authoritarian regime breakdown and democratic transition. Comments and questions are encouraged. 

2 September 2015

- `space_time_lags.ipynb`: Notebook demonstrating the process for generating spatiotemporal lags of regime failure. Replaces earlier code. 
- `split_population_logit.py`: Class and functions for estimating the split population logistic regression with optional regularization parameters. 

30 June 2015

- `separation_plot.py`: Python implementation of the separation plot, described in [Greenhill, B., Ward, M. D. and Sacks, A. (2011)](http://onlinelibrary.wiley.com/doi/10.1111/j.1540-5907.2011.00525.x/abstract;jsessionid=BD5CAFFC29F5F6226ECCC31EE41A0CCB.f03t04?deniedAccessCustomisedMessage=&userIsAuthenticated=false), using Pandas and Matplotlib. 

29 June 2015

- `dissertation_programs.py`: Programs for estimating and presenting results for dissertation chapter 1 analysis.

16 December 2014

- `observed_case_simulation.R`: this script contains programs for simulated of predicted probabilities for logistic regression models and generalized additive models for binary dependent variables using the observed-case method. A short demonstration is available [here](http://www.thomaswbrawner.com/simulation.html 'Simulation demo').

5 December 2014

- `linguistic_distance.py`: this script scrapes the principal languages for all countries in the world from [Ethnologue](http://www.ethnologue.com/ 'Ethnologue: Languages of the World'), including statutory languages and de facto languages of national identity. In turn, it generates a measure of binary connectivity for all country dyads in the world, taking a value of one where there is an intersection in the principal languages of the respective countries and zero otherwise. This script relies on `country_codes.py`. 
- `country_codes.py`: this script includes a dictionary mapping country names to Correlates of War country codes as well as a function to generate the country codes for a provided *pandas* data frame. 

22 November 2014

- `kfold_cv_programs.R`: this is a collection of programs for conducting k-fold cross-validation, iterating k-fold cross-validation, and visualizing predictive performance. A short demo is available [here](http://www.thomaswbrawner.com/cross-validation.html 'k-fold CV demo').
 


