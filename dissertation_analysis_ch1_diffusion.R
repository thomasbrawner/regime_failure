## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##
## dissertation_analysis_ch2_performance.R
## plot the estimated predicted probabilities from democratic diffusion model
## tb 16 mar 2015
## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##

library(ggplot2)

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##
## read the data

data = read.csv('diffusion/year_probabilities.csv')

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##
## plot probabilities over years -- scatter with jitter, smoothing

p = ggplot(data, aes(year, probs)) + 
	geom_point(position = 'jitter', alpha = 0.4, size = 1) + 
	stat_smooth(se = FALSE, size = 1) + 
	labs(x = '', y = 'Pr(Democratic Transition)') + 
	theme(axis.title.y = element_text(size = 10),
		  axis.title.x = element_text(size = 10),
		  axis.text.y = element_text(colour = 'black'),	
		  axis.text.x = element_text(colour = 'black')) 

pdf('diffusion/figures/probabilities.pdf')
print(p)
dev.off()

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##
## plot probabilities over years -- boxplots

p = ggplot(data, aes(factor(year), probs)) + 
	geom_boxplot() + 
	labs(x = '', y = 'Pr(Democratic Transition)') + 
	theme(axis.title.y = element_text(size = 10),
		  axis.title.x = element_text(size = 10),
		  axis.text.y = element_text(colour = 'black'),	
		  axis.text.x = element_text(colour = 'black', angle = 45, hjust = 1, vjust = 1)) 

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##
## plot probabilities over 5-year periods -- boxplots

p = ggplot(data, aes(factor(lustrum), probs)) + 
	geom_boxplot() +  
	labs(x = '', y = 'Pr(Democratic Transition)') + 
	theme(axis.title.y = element_text(size = 10),
		  axis.title.x = element_text(size = 10),
		  axis.text.y = element_text(colour = 'black'),	
		  axis.text.x = element_text(colour = 'black')) 

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##
## plot first differences over years -- scatter with jitter, smoothing

p = ggplot(data, aes(year, fd)) + 
	geom_point(position = 'jitter', alpha = 0.4, size = 1) + 
	stat_smooth(se = FALSE, size = 1) + 
	labs(x = '', y = expression(paste(Delta, ' Pr(Democratic Transition)'))) + 
	theme(axis.title.y = element_text(size = 10),
		  axis.title.x = element_text(size = 10),
		  axis.text.y = element_text(colour = 'black'),	
		  axis.text.x = element_text(colour = 'black')) 

pdf('diffusion/figures/first_differences.pdf')
print(p)
dev.off()

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##
## plot first differences over 5-year periods -- boxplots

p = ggplot(data, aes(factor(year), fd)) + 
	geom_boxplot() +
	labs(x = '', y = expression(paste(Delta, ' Pr(Democratic Transition)'))) + 
	theme(axis.title.y = element_text(size = 10),
		  axis.title.x = element_text(size = 10),
		  axis.text.y = element_text(colour = 'black'),	
		  axis.text.x = element_text(colour = 'black')) 

## ------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- ##
