library(Amelia)

data = read.table('clean_data/full_regime_data_to_impute.txt', sep=',', header=TRUE)
imputation_model = amelia(data, m=10, idvars=c('date'), p2s=2, empri=(0.5 * nrow(data)), incheck=FALSE)
write.amelia(obj=imputation_model, file.stem='clean_data/imputation_')
saveRDS(imputation_model, 'clean_data/imputation_object.rds')

