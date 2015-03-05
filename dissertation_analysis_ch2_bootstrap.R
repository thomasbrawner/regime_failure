## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##
## dissertation_analysis_ch2_performance.R
## plot the estimated predicted probabilities for counterfactual values of spatial lag, estimates from sklearn classifiers
## tb 26 feb 2015
## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##

library(ggplot2)
library(plyr)
library(reshape2)

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##

marginal_plots = function(data){
	data$lag_value = round(data$lag_value, 2)
	p = ggplot(data, aes(factor(lag_value), median_prob)) + 
			   geom_boxplot() + 
			   facet_wrap(~ period, ncol = 3) + 
			   labs(x = 'Spatiotemporal Lag', y = 'Predicted Probability of Failure') + 
			   theme(axis.title.y = element_text(size = 10),
			   		 axis.title.x = element_text(size = 10),
		  	   		 axis.text.y = element_text(colour = 'black'),	
		  	   		 axis.text.x = element_text(colour = 'black', angle = 45, hjust = 1, vjust = 1)) 
	print(p)
}

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##

data = read.csv('classification/bootstrap/logit_failure_failure.csv')
data = data[which(data$period != 1945),]
data = data[which(data$period != 2010),]

pdf('classification/bootstrap/probabilities_logit_failure_failure.pdf')
marginal_plots(data)
dev.off()

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##

data = read.csv('classification/bootstrap/logit_coerce_failure.csv')
data = data[which(data$period != 1945),]
data = data[which(data$period != 2010),]

pdf('classification/bootstrap/probabilities_logit_coerce_failure.pdf')
marginal_plots(data)
dev.off()

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##

data = read.csv('classification/bootstrap/logit_autocratic_failure.csv')
data = data[which(data$period != 1945),]
data = data[which(data$period != 2010),]

pdf('classification/bootstrap/probabilities_logit_autocratic_failure.pdf')
marginal_plots(data)
dev.off()

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##

data = read.csv('classification/bootstrap/lasso_failure_failure.csv')
data = data[which(data$period != 1945),]
data = data[which(data$period != 2010),]

pdf('classification/bootstrap/probabilities_lasso_failure_failure.pdf')
marginal_plots(data)
dev.off()

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##

data = read.csv('classification/bootstrap/lasso_coerce_failure.csv')
data = data[which(data$period != 1945),]
data = data[which(data$period != 2010),]

pdf('classification/bootstrap/probabilities_lasso_coerce_failure.pdf')
marginal_plots(data)
dev.off()

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##

data = read.csv('classification/bootstrap/lasso_autocratic_failure.csv')
data = data[which(data$period != 1945),]
data = data[which(data$period != 2010),]

pdf('classification/bootstrap/probabilities_lasso_autocratic_failure.pdf')
marginal_plots(data)
dev.off()

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##

data = read.csv('classification/bootstrap/random_forest_failure_failure.csv')
data = data[which(data$period != 1945),]
data = data[which(data$period != 2010),]

pdf('classification/bootstrap/probabilities_random_forest_failure_failure.pdf')
marginal_plots(data)
dev.off()

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##

data = read.csv('classification/bootstrap/random_forest_failure_coerce.csv')
data = data[which(data$period != 1945),]
data = data[which(data$period != 2010),]

pdf('classification/bootstrap/probabilities_random_forest_failure_coerce.pdf')
marginal_plots(data)
dev.off()

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##

data = read.csv('classification/bootstrap/random_forest_failure_autocratic.csv')
data = data[which(data$period != 1945),]
data = data[which(data$period != 2010),]

pdf('classification/bootstrap/probabilities_random_forest_failure_autocratic.pdf')
marginal_plots(data)
dev.off()

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##

data = read.csv('classification/bootstrap/random_forest_coerce_coerce.csv')
data = data[which(data$period != 1945),]
data = data[which(data$period != 2010),]

pdf('classification/bootstrap/probabilities_random_forest_coerce_coerce.pdf')
marginal_plots(data)
dev.off()

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##

data = read.csv('classification/bootstrap/random_forest_autocratic_coerce.csv')
data = data[which(data$period != 1945),]
data = data[which(data$period != 2010),]

pdf('classification/bootstrap/probabilities_random_forest_autocratic_coerce.pdf')
marginal_plots(data)
dev.off()

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##
