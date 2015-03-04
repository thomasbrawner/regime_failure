## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##
## dissertation_analysis_ch2_performance.R
## tb 25 feb 2015, last update 2 mar 2015
## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##

rm(list = ls())

library(ggplot2)
library(plyr)
library(reshape2)

## ------------------------------------------------------------------------------- ##
## function to clean up levels in the performance data

fix_labels = function(data, interaction = FALSE){
	if(interaction){
		data$period = ifelse(data$period == '', 'no period effects',
			  	  	  ifelse(data$period == 'lag_lustrum', '5-year effects X lag',
			  	  	  ifelse(data$period == 'lustrum', '5-year effects', '2-year effects')))
		data$period = factor(data$period, c('no period effects','2-year effects','5-year effects','5-year effects X lag'), ordered = TRUE)
	} else {
		data$period = ifelse(data$period == '', 'no period effects',
			  	  	  ifelse(data$period == 'lustrum', '5-year effects', '2-year effects'))
		data$period = factor(data$period, c('no period effects','2-year effects','5-year effects'), ordered = TRUE)
	}

	data$model = ifelse(data$model == 0, 'no unit effects',
			 	 ifelse(data$model == 1, 'region effects', 'country effects'))
	data$model = factor(data$model, c('no unit effects','region effects','country effects'), ordered = TRUE)	
	
	return(data)
}

## ------------------------------------------------------------------------------- ##
## functions to melt performance data for faceted performance plots

metric_melt = function(data, metric){
	metric_columns = grep(metric, names(data), value = TRUE)
	data = data[,c('dependent','model','period',metric_columns)]
	data = melt(data, id.vars = c('dependent','model','period'), variable.name = 'lag', value.name = metric)
	data$lag = as.character(data$lag)
	data$lag = matrix(unlist(strsplit(data$lag, '_')), ncol = 2, byrow = TRUE)[,2]
	data$lag = factor(data$lag, c('restricted','failure','coerce','autocratic','democratic'), ordered = TRUE)
	return(data)
}

## ------------------------------------------------------------------------------- ##
## delta performance data between restricted and unrestricted specifications

metric_melt_delta = function(data, metric){
	metric_columns = grep(metric, names(data), value = TRUE)
	restricted_column = paste(metric, 'restricted', sep = '_')
	data$delta_failure = data[,paste(metric, 'failure', sep = '_')] - data[,restricted_column]
	data$delta_coerce = data[,paste(metric, 'coerce', sep = '_')] - data[,restricted_column]
	data$delta_autocratic = data[,paste(metric, 'autocratic', sep = '_')] - data[,restricted_column]
	data$delta_democratic = data[,paste(metric, 'democratic', sep = '_')] - data[,restricted_column]
	metric_columns = grep('delta', names(data), value = TRUE)
	data = data[,c('dependent','model','period',metric_columns)]
	data = melt(data, id.vars = c('dependent','model','period'), variable.name = 'lag', value.name = metric)
	data$lag = as.character(data$lag)
	data$lag = matrix(unlist(strsplit(data$lag, '_')), ncol = 2, byrow = TRUE)[,2]
	data$lag = factor(data$lag, c('failure','coerce','autocratic','democratic'), ordered = TRUE)
	return(data)
}

## ------------------------------------------------------------------------------- ##
## function to output the faceted performance plots for each dependent variable

performance_plots = function(data, classifier, metric = 'auroc'){
	dep_vars = c('failure','coerce','auttrans','demtrans')
	for(dep_var in dep_vars){
		df = data[which(data[,'dependent'] == dep_var),]
	
		# melt to compare restricted to full specifications on metric
		df1 = metric_melt(df, metric)
		
		# plot AUROC by restricted vs full
		pdf(paste0('classification/performance_figures/performance_', metric, '_', classifier, '_', dep_var, '.pdf'))
		p = ggplot(df1, aes_string('lag', metric)) + 
			   	   geom_boxplot() +
			   	   facet_grid(period ~ model) + 
			   	   labs(x = '', y = '') + 
			       theme(axis.title.y = element_text(size = 10),
			   		     axis.title.x = element_text(size = 10),
		  	   		 	 axis.text.y = element_text(colour = 'black'),	
		  	   		 	 axis.text.x = element_text(colour = 'black', angle = 45, hjust = 1, vjust = 1)) 
		print(p)
		dev.off()
		
		# evaluate delta metric and melt to plot
		df2 = metric_melt_delta(df, metric)
		
		# plot delta 
		pdf(paste0('classification/performance_figures/performance_', metric, '_', classifier, '_', dep_var, '_delta.pdf'))
		p = ggplot(df2, aes_string('lag', metric)) + 
			   	   geom_boxplot() + 
			   	   geom_hline(yintercept = 0, linetype = 2) + 
			   	   facet_grid(period ~ model) + 
			   	   labs(x = '', y = '') + 
			   	   theme(axis.title.y = element_text(size = 10),
			   		 	 axis.title.x = element_text(size = 10),
		  	   		 	 axis.text.y = element_text(colour = 'black'),	
		  	   		 	 axis.text.x = element_text(colour = 'black', angle = 45, hjust = 1, vjust = 1)) 
		print(p)
		dev.off()   
		}
}

## ------------------------------------------------------------------------------- ##
## lasso

data = read.csv('classification/performance_lasso.csv')
data = fix_labels(data, interaction = TRUE)
performance_plots(data, 'lasso')
performance_plots(data, 'lasso', 'recall')

## ------------------------------------------------------------------------------- ##
## logit

data = read.csv('classification/performance_logit.csv')
data = fix_labels(data, interaction = TRUE)
performance_plots(data, 'logit')
performance_plots(data, 'logit', 'recall')

## ------------------------------------------------------------------------------- ##
## svm

data = read.csv('classification/performance_svm.csv')
data = fix_labels(data, interaction = TRUE)
performance_plots(data, 'svm')
performance_plots(data, 'svm', 'recall')

## ------------------------------------------------------------------------------- ##
## random forest

data = read.csv('classification/performance_random_forest.csv')
data = fix_labels(data, interaction = TRUE)
performance_plots(data, 'random_forest')
performance_plots(data, 'random_forest', 'recall')

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##
