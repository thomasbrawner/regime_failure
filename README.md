## Regime failure

This repository will host several Python and R scripts for data management and analysis related to the breakdown of authoritarian regimes. Additions are made as they become available. Comments and questions are encouraged. 

15 February 2015

- `dissertation_analysis_ch1b.py`: generates plots of simulated first differences for a shift in the value of spatiotemporal lags conditional on five-year time periods over time. Relies on `dissertation_programs.py`. 

15 February 2015

- `dissertation_analysis_ch1a.py`: generates regression tables and plots of simulated first differences demonstrating the estimated effect of spatiotemporal lags of various categories of authoritarian regime failure on authoritarian regime stability. Relies on the programs in `dissertation_programs.py`.

3 February 2015

- `dissertation_programs.py`: updates to programs for dissertation analysis. Include complementary functions for plotting first differences in simulated quantities of interest, for generating LaTeX tables from estimation output, and other functions for data wrangling and transformation.

22 January 2015

- `dissertation_programs.py`: this Python script generates a class from a *pandas* data frame suitable for data analysis. The class has methods for parameter estimation, simulation of quantities of interest, and classification for binary dependent variable models. 

5 January 2015

- `spatiotemporal_lags.py`: this Python script replaces earlier code for the construction of spatiotemporal lags of authoritarian regime failure. Greater explanation is available [here](http://www.thomaswbrawner.com/spatiotemporal-lags.html 'Spatial lags explanation').

16 December 2014

- `observed_case_simulation.R`: this script contains programs for simulated of predicted probabilities for logistic regression models and generalized additive models for binary dependent variables using the observed-case method. A short demonstration is available [here](http://www.thomaswbrawner.com/simulation.html 'Simulation demo').

5 December 2014

- `linguistic_distance.py`: this script scrapes the principal languages for all countries in the world from [Ethnologue](http://www.ethnologue.com/ 'Ethnologue: Languages of the World'), including statutory languages and de facto languages of national identity. In turn, it generates a measure of binary connectivity for all country dyads in the world, taking a value of one where there is an intersection in the principal languages of the respective countries and zero otherwise. This script relies on `country_codes.py`. 
- `country_codes.py`: this script includes a dictionary mapping country names to Correlates of War country codes as well as a function to generate the country codes for a provided *pandas* data frame. 

22 November 2014

- `kfold_cv_programs.R`: this is a collection of programs for conducting k-fold cross-validation, iterating k-fold cross-validation, and visualizing predictive performance. A short demo is available [here](http://www.thomaswbrawner.com/cross-validation.html 'k-fold CV demo').
 


