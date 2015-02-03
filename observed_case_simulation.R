## --------------------------------------------------------------------------------- ## 
## --------------------------------------------------------------------------------- ## 
## observed_case_simulation.R
## simulate & plot predicted values from logit/GAM for binary dependent variable
## using the observed-case method
## thomas brawner, 16 December 2014
## --------------------------------------------------------------------------------- ## 
## --------------------------------------------------------------------------------- ## 

require(boot)
require(ggplot2)
require(MASS)

## --------------------------------------------------------------------------------- ## 
## bootstrap mean

boot_mean = function(data, indices){
    d = data[indices]
    mu = mean(d)
    return(mu)
}

## --------------------------------------------------------------------------------- ## 
## observed case simulation for logit

ocsim_logit = function(x, var_name, values = NULL, nsims = 1000, prob_scale = TRUE){
    
    model_frame = x$model
    model_frame[,1] = 1

    if(!is.null(values)){
        values = values
    } else if(length(unique(model_frame[,var_name])) < 20){
        values = unique(model_frame[,var_name])
    } else {
        values = seq(min(model_frame[,var_name]), max(model_frame[,var_name]), length.out = 20)
    }

    bsims = mvrnorm(nsims, coef(x), vcov(x))
    val = c()
    med = c()
    upp = c()
    low = c()
    
    for(value in values){
	preds = matrix(NA, nrow = nrow(model_frame), ncol = nsims)
        model_frame[,var_name] = value
	for(b in 1:nrow(bsims)){
	    preds[,b] = as.matrix(model_frame) %*% bsims[b,]
	}
        out = unlist(lapply(apply(preds, 2, boot, boot_mean, R = 100), function(x) x$t))
	med = c(med, median(out))
	upp = c(upp, quantile(out, prob = .975))
	low = c(low, quantile(out, prob = .025))
    	val = c(val, value)
    }
   
    if(prob_scale){
        med = (1 / (1 + exp(-med)))
	upp = (1 / (1 + exp(-upp)))
	low = (1 / (1 + exp(-low)))
	return(data.frame(cbind(val, med, upp, low)))
    } else {
        return(data.frame(cbind(val, med, upp, low)))
    }
}

## --------------------------------------------------------------------------------- ## 
## observed case simulation for gam

ocsim_gam = function(x, model_frame, var_name, values = NULL, nsims = 1000, prob_scale = TRUE){
    
    if(!is.null(values)){
        values = values
    } else if(length(unique(model_frame[,var_name])) < 20){
        values = unique(model_frame[,var_name])
    } else {
        values = seq(min(model_frame[,var_name]), max(model_frame[,var_name]), length.out = 20)
    }

    bsims = mvrnorm(nsims, coef(x), vcov(x))
    val = c()
    med = c()
    upp = c()
    low = c()
    
    for(value in values){
	preds = matrix(NA, nrow = nrow(model_frame), ncol = nsims)
        model_frame[,var_name] = value
	xp = predict(x, model_frame, type = 'lpmatrix')
	for(b in 1:nrow(bsims)){
	    preds[,b] = xp %*% bsims[b,]
	}
        out = unlist(lapply(apply(preds, 2, boot, boot_mean, R = 100), function(x) x$t))
	med = c(med, median(out))
	upp = c(upp, quantile(out, prob = .975))
	low = c(low, quantile(out, prob = .025))
    	val = c(val, value)
    }
   
    if(prob_scale){
        med = (1 / (1 + exp(-med)))
	upp = (1 / (1 + exp(-upp)))
	low = (1 / (1 + exp(-low)))
	return(data.frame(cbind(val, med, upp, low)))
    } else {
        return(data.frame(cbind(val, med, upp, low)))
    }
}

## --------------------------------------------------------------------------------- ## 
## plot results from simulation

plot_ocsim = function(ocsim_output, x_label = NULL, y_label = NULL){
    p = ggplot(ocsim_output, aes(x = val, y = med)) + 
        geom_line(colour = 'black') + 
	geom_ribbon(aes(ymin = low, ymax = upp), alpha = .2) + 
	labs(x = x_label, y = y_label) + 
	theme(axis.title.x = element_text(size = 10), 
              axis.title.y = element_text(size = 10))
    print(p)
}

## --------------------------------------------------------------------------------- ## 
## --------------------------------------------------------------------------------- ## 
