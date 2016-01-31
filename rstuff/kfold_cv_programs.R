# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #
# kfold_cv_programs.R
# functions to perform k-fold cross-validation for classification models
# tb 22 nov 2014, last update 15 feb 2015
# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #

require(caret)
require(ggplot2)
require(graphics)
require(grDevices)
require(grid)
require(pROC)
require(wq)

# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #
# k-fold cross-validation
# arguments:
# -- formula: as.formula(), for use with logit, gam, randomForest
# -- data: data frame (no NA)
# -- depvar: string -- 'depvar'
# -- k: k folds (integer)
# -- method: choose classifier
# -- covariates: predictor variables, for use with krls
# -- seed: optional seed for replication
# values:
# -- a data frame with the observed outcomes and the predicted probabilities

kfoldcv = function(formula = NULL, data, depvar, k, method = c('krls','logit','logit.gam','randomForest'), 
                   covariates = NULL, seed = NULL){
  if(any(is.na(data))){
    stop('There are NA observations in the data frame. Please remove them.')
  }
  if(!is.null(seed)){
    set.seed(seed)
  } 
  
  data[,'fold'] = createFolds(data[,depvar], k = k, list = FALSE)
  predictions = data.frame()
  
  for(f in 1:k){
    
    test_data = data[which(data$fold == f), !(names(data) %in% c('fold'))]
    train_data = data[which(data$fold != f), !(names(data) %in% c('fold'))]
    
    if(method == 'logit') {
      model = glm(formula, data = train_data, family = 'binomial')
      test_data$prediction = predict(model, test_data, type = 'response')
    } else if(method == 'logit.gam') {
      require(mgcv)
      model = gam(formula, data = train_data, family = 'binomial')
      test_data$prediction = predict(model, test_data, type = 'response')
    } else if(method == 'randomForest') {
      require(randomForest)
      model = randomForest(formula, data = train_data, ntree = 1000)
      test_data$prediction = predict(model, test_data, type = 'prob')[,2]
    } else if(method == 'krls') {
      require(KRLS)
      train_x = train_data[,which(names(train_data) %in% covariates)]
      test_x = test_data[,which(names(test_data) %in% covariates)]
      model = krls(X = train_x, y = train_data[,depvar])
      test_data$prediction = predict(model, test_x)$fit
    } else {
      stop('Method not supported.')
    }
    
    test_data$row = rownames(test_data)
    predictions = rbind(predictions, data.frame(test_data[,c('prediction','row')]))
  }
  
  d = data
  d$row = rownames(data)
  d = merge(d, predictions, by = 'row')
  names(d)[which(names(d) == depvar)] = 'class'
  return(d[,c('class','prediction')])
}

# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #
# iterations of k-fold cross-validation 
# arguments:
# -- iterations: how many iterations? 
# -- arguments to kfoldcv
# -- seed: for replication purposes
# values:
# -- predictions: a data frame with the observed outcomes and the mean of predicted probabilities for
#    each observation across the n iterations. 
# -- predictions_iterations: predicted probabilities by observation (row) across the iterations (column)
# -- seeds: seeds fed to createFolds in kfoldcv for splitting the data frame

itercv = function(iterations = 10, formula, data, depvar, k, method, seed = NULL){
  if(!is.null(seed)){
    set.seed(seed)
  } 
  seeds = as.list(sample(1:1000000, iterations))
  output_list = lapply(seeds, function(x) kfoldcv(formula, data, depvar, k, method, seed = x))
  predictions_list = lapply(output_list, function(x) x[,2])
  predictions_mat = do.call(cbind, predictions_list)
  predictions = rowMeans(predictions_mat)
  observations = output_list[[1]][,1]
  return(list('predictions' = data.frame('class' = observations, 'prediction' = predictions),
              'predictions_iterations' = predictions_mat,
              'seeds' = unlist(seeds)))
}

# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #
# predictive performance plot
# arguments: 
# -- x: data frame returned by kfoldcv() or itercv()$predictions
# -- print_auc: should the AUC be printed?
# -- print_ci_auc: should the 95% confidence interval for the AUC be printed?
# values: 
# -- ROC curve
# -- boxplots, distributions of predicted probabilities by class

plot_kfoldcv = function(x, print_auc = TRUE, print_ci_auc = TRUE, distribution = TRUE){    
  
  curve = roc(as.factor(x[,1]), x[,2])
  auroc_lb = paste('AUC[l] == ', format(round(ci.auc(roc(as.factor(x[,1]), x[,2]))[1], 3), nsmall = 3))
  auroc_mu = paste('AUC[mu] == ', format(round(ci.auc(roc(as.factor(x[,1]), x[,2]))[2], 3), nsmall = 3))
  auroc_ub = paste('AUC[u] == ', format(round(ci.auc(roc(as.factor(x[,1]), x[,2]))[3], 3), nsmall = 3))
  
  curve = data.frame(cbind(1 - curve$specificities), curve$sensitivities)
  names(curve) = c('fpr','tpr')
  
  p = ggplot(curve, aes(x = fpr, y = tpr)) + geom_line() + 
    scale_y_continuous(breaks = c(0, .2, .4, .6, .8, 1), limits = c(0,1)) + 
    scale_x_continuous(breaks = c(0, .2, .4, .6, .8, 1), limits = c(0,1)) + 
    labs(x = 'false positive rate', y = 'true positive rate') + 
    geom_abline(intercept = 0, slope = 1, linetype = 3, alpha = 0.8) + 
    theme(axis.title.x = element_text(size = 10), 
          axis.title.y = element_text(size = 10))
  
  if(print_auc){
    p = p + annotate('text', x = .8, y = .16, label = auroc_mu, parse = TRUE, size = 3.5)
  }
  
  if(print_ci_auc){
    p = p + annotate('text', x = .8, y = .24, label = auroc_ub, parse = TRUE, size = 3.5) + 
            annotate('text', x = .8, y = .08, label = auroc_lb, parse = TRUE, size = 3.5)
  }
  
  x[,1] = as.factor(x[,1])
  d = ggplot(x, aes_string(names(x)[1], names(x)[2])) + geom_boxplot() + 
    labs(x = 'class', y = 'predicted probability') + 
    theme(axis.title.x = element_text(size = 10), 
          axis.title.y = element_text(size = 10))
  
  return(layOut(list(p, 1, 1),
                list(d, 1, 2)))
}

# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #
