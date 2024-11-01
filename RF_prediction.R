rm(list = ls())

library(ranger)
library(dplyr)
library(scoringRules)
library(quantregForest)
library(knitr)

set.seed(2024)

# set working directory
setwd("/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/R_code")

if (!dir.exists("res")) {
  dir.create("res")
}


# which implementation to use? 
which_package <- "ranger" # "ranger" or "quantregForest"
bagged_trees <- TRUE # if true, use bagged trees (without subsampling regressors)

# read training data
dat <- read.csv("/Users/sophiasiefert/Documents/Vorlesungen /Master/Masterarbeit/Data/rf_data_1823_clean.csv") %>%
  mutate(date = as.Date(date))

n_trees <- 500 # number of trees to be used
n <- 35056 # size of training sample (2018-2021)
dat_train <- dat[1:n, ] # training data
dat_test <- dat[(n+1):nrow(dat), ] # test data
y_train <- dat_train$load
y_test <- dat_test$load

# remove outcome from test data, just to be sure
dat_test <- dat_test %>% select(-load)

# show range of training sample
range(dat_train$date)
# show dimension of training sample (35056, 16)
dim(dat_train)

# show range of testing sample
range(dat_test$date)
# show dimension of testing sample (16449, 15)
dim(dat_test)

# plot data ----
plot(dat$date, dat$load, bty = "n", xlab = "Date", ylab = "Load", 
     type = "l")
# end of training sample marked in green
abline(v = tail(dat_train$date, 1), col = "green4", lwd = 3)



# settings for random forest ---
time_trend <- TRUE # if false, omit time trend
day_of_year <- FALSE # if false, use sparser (monthly) coding

# Number of quantiles to be used
n_quantiles <- 1e2

# Grid of quantiles (see Barbiero & Hitaj, 2023, Equation 4)
grid_quantiles <- (2*(1:n_quantiles)-1)/(2*n_quantiles)

# construct formula
fml <- "load ~ holiday + hour_int + weekday_int"

if (day_of_year){
  fml <- paste0(fml, "+ yearday")
} else {
  fml <- paste0(fml, "+ month_int")
}
if (time_trend){
  fml <- paste0(fml, " + time_trend")
}

fml <- as.formula(fml)

# fit model ---
# fit model
if (which_package == "ranger"){
  m_try <- ifelse(bagged_trees, identity, 
                  function(d) floor(d/3))
  fit <- ranger(fml, data = dat_train, mtry = m_try,
                quantreg = TRUE, keep.inbag = TRUE, 
                num.trees = n_trees)  
  # check 
  stopifnot(fit$num.trees == n_trees)
  pred <- predict(fit, type = "quantiles", 
                  quantiles = grid_quantiles, 
                  data = dat_test)$predictions
} else {
  x_train <- model.matrix(fml, data = dat_train) %>%
    as.matrix
  x_test <- model.matrix(fml, data = data.frame(load = y_test, 
                                                dat_test)) %>%
    as.matrix
  # check
  stopifnot(all(colnames(x_test) != "load"))
  # use mtry = ncol(x_train) for bagged trees
  # otherwise (per default), mtry = sqrt(p) is used
  m_try <- ifelse(bagged_trees, ncol(x_train), 
                  floor(ncol(x_train)/3))
  fit <- quantregForest(x = x_train, y = as.matrix(y_train), 
                        ntree = n_trees, 
                        mtry = m_try)
  stopifnot(fit$ntree == n_trees)
  pred <- predict(fit, what = grid_quantiles, 
                  newdata = x_test)
}

# compute crps
res <- data.frame(date_time = dat_test$date_time,
                  crps = NA, ae = NA, se = NA, 
                  year = dat_test$year)
# loop over test sample observations
for (jj in 1:nrow(dat_test)){
  res$crps[jj] <- crps_sample(y = y_test[jj], 
                              dat = pred[jj, ])
  res$ae[jj] <- abs(y_test[jj]-median(pred[jj,]))
  res$se[jj] <- (y_test[jj]-mean(pred[jj,]))^2
}

res %>% group_by(year) %>%
  summarise(crps = mean(crps), ae = mean(ae), se = mean(se)) %>%
  kable(digits = 1)

# save results in subfolder "res/"
save_name <- paste0("res/", which_package, "_")
if (time_trend){
  save_name <- paste0(save_name, "tt_")
} else {
  save_name <- paste0(save_name, "nott_")
}
if (day_of_year){
  save_name <- paste0(save_name, "day_")
} else {
  save_name <- paste0(save_name, "month_")
}
if (bagged_trees){
  save_name <- paste0(save_name, "bt.csv")
} else {
  save_name <- paste0(save_name, "rf.csv")
}

write.table(res, file = save_name, 
            sep = ",", row.names = FALSE)

# Overall Results (crps and mse)
overall_results <- res %>%
  summarise(mean_crps = mean(crps, na.rm = TRUE),  # average CRPS
            mean_ae = mean(ae, na.rm = TRUE),      # average AE (MAE)
            mean_se = mean(se, na.rm = TRUE),      # average SE (MSE)
            root_mse = sqrt(mean_se))              # root mse

# Display results
overall_results %>%
  kable(digits = 4)



