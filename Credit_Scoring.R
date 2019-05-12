# required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(Hmisc)) install.packages("Hmisc", repos = "http://cran.us.r-project.org")
if(!require(smbinning)) install.packages("smbinning", repos = "http://cran.us.r-project.org")
if(!require(doParallel)) install.packages("doParallel", repos = "http://cran.us.r-project.org")
if(!require(ResourceSelection)) install.packages("ResourceSelection", repos = "http://cran.us.r-project.org")
if(!require(devtools)) install.packages("devtools", repos = "http://cran.us.r-project.org")
if(!require(rlang)) install.packages("rlang", repos = "http://cran.us.r-project.org")

devtools::install_github("shichenxie/scorecard", force = FALSE)
devtools::install_github("ayhandis/creditR", force = FALSE)

library(tidyverse)
library(caret)
library(scorecard)  # creates a scorecard
source('smbinning.R')  # Optimal Binning categorizes a numeric characteristic into bins 
# for ulterior usage in scoring modeling.
library(doParallel) # Solve 'Error in summary.connection(connection) : invalid connection'
# https://stackoverflow.com/questions/25097729/un-register-a-doparallel-cluster

library(ResourceSelection) # Compute Hosmer Lemeshow test
library(Hmisc)             # Compute Gini and ROC index
library(creditR)           # Compute Binomial tests

# load data
data <- read.csv("C:/Users/Mario/OneDrive/Escritorio/data.csv", sep=";")
str(data)
dim(data)

#ind <- grep(pattern = '^situ',x = names(data))
#data[,ind] <- lapply(data[,ind] , factor)
data$repartition <- factor(data$repartition)

data <- data %>%
  mutate(reclassified = ifelse(reclassified == 'yes',1L,0L)) %>%
  mutate(tax_identification_number = floor(tax_identification_number / 10^9)) %>%
  mutate(sex = factor(ifelse(tax_identification_number == 27 | tax_identification_number == 23,
                              'Female', 'Male'))) %>%
  # mutate(maximum_situation_12_months = factor(maximum_situation_12_months)) %>%
  mutate(reclassified_prior_11_months = factor(ifelse(reclassified_prior_11_months == '0','no','yes'))) %>% 
  #mutate(quantity_debt_financial_institution_no_bank = factor(quantity_debt_financial_institution_no_bank)) %>%
  select(-tax_identification_number)

# Optimal bins
opt_bins <- smbinning.sumiv(data, y = 'reclassified')
opt_bins %>% knitr::kable()

no_sig <- opt_bins %>% filter(is.na(IV))
data <- data %>%
  select(-c( as.character(no_sig$Char)))

# Optimal bins
opt_bins <- smbinning.sumiv(data, y = 'reclassified')
opt_bins %>% knitr::kable()
smbinning.sumiv.plot(opt_bins, cex = 0.9)

result <- smbinning.factor(data, 'reclassified_prior_11_months', y = 'reclassified')
result$ivtable %>% knitr::kable()
result$iv # Information value

par(mfrow=c(2,2))
smbinning.plot(result,option="dist",sub="reclassified_prior_11_months")
smbinning.plot(result,option="badrate",sub="reclassified_prior_11_months")
smbinning.plot(result,option="WoE",sub="reclassified_prior_11_months")
par(mfrow=c(1,1))

result <- smbinning.monotonic(data, x = 'average_debt_months_different_situation_1', y = 'reclassified')
result$ivtable %>% knitr::kable()
result$iv # Information value
result$bands # Bins or bands


par(mfrow=c(2,2))
boxplot(data$average_debt_months_different_situation_1~data$reclassified,
        horizontal=TRUE, frame=FALSE, col="lightgray",main="Distribution")
mtext("average_debt_months_different_situation_1",3)
smbinning.plot(result,option="dist",sub="average_debt_months_different_situation_1")
smbinning.plot(result,option="badrate",sub="average_debt_months_different_situation_1")
smbinning.plot(result,option="WoE",sub="average_debt_months_different_situation_1")
par(mfrow=c(1,1))

data_opt <- data %>% 
  select(as.character(opt_bins$Char [opt_bins$IV >= 0.02]), reclassified)

class <- lapply(data_opt, class)
ind <- which(class == 'factor')

bins_num <- lapply(names(data_opt) [-c(ind,length(class))], function(x) smbinning.monotonic(df = data_opt, 
                                                                                  y = 'reclassified',
                                                                                  x = x, p = 0.01)) 


bins_factor <- lapply(names(data_opt) [ind], function(x) smbinning.factor(df = data_opt, 
                                                                          y = 'reclassified',
                                                                          x = x)) 

#Traditional Credit Scoring Using Logistic Regression
# breaking dt into train and test
seed <- 123
keep <- names(data_opt) [-length(data_opt)]
dt_f <- var_filter(data_opt, y = 'reclassified', positive = 1, var_kp = keep)
dt_list <- split_df(dt_f, y = 'reclassified', ratio = 0.7, seed = seed)
label_list <- lapply(dt_list, function(x) x$reclassified)

breaks_adj <- list()

for (i in 1:length(x = bins_factor)) {
  var_name <- bins_factor[[i]]$x
  append(breaks_adj, breaks_adj[[var_name]] <- bins_factor[[i]]$cuts)
}

for (i in 1:length(x = bins_num)) {
  var_name <- bins_num[[i]]$x
  append(breaks_adj, breaks_adj[[var_name]] <- bins_num[[i]]$cuts)
}

bins_adj <- suppressWarnings(woebin(dt_f, y = "reclassified", breaks_list = breaks_adj, 
                                    var_kp = keep, save_breaks_list = NULL))

# converting train and test into woe values
dt_woe_list = lapply(dt_list, function(x) suppressWarnings(woebin_ply(x, bins_adj)))

# Select variables by Regularization

cl <- makeCluster(2)
registerDoParallel(cl)
on.exit(stopCluster(cl))

# define training control
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

# Select variates using cv and glmnet

set.seed(seed)
m1 <- train(factor(reclassified) ~ .,
            data = dt_woe_list$train,
            trControl = train_control,
            method = "glmnet",
            family= 'binomial')

m1_reg <- coef(m1$finalModel, m1$bestTune$lambda) [-1,1] != 0

cols_not_remove <-  str_remove(c(names(coef(m1$finalModel, m1$bestTune$lambda) [-1,1])),'_woe') [m1_reg]
y <- dt_f$reclassified
cols <- names(dt_f)   #paste(names(dt_f),'_woe', sep='')
dt_f <- dt_f %>% select(match(cols_not_remove, cols))
dt_f <- dt_f %>% mutate(reclassified = y)

class <- lapply(dt_f, class)
ind <- which(class == 'factor')

bins_num <- lapply(names(dt_f) [-c(ind,length(class))], function(x) smbinning.monotonic(df = dt_f, 
                                                                                            y = 'reclassified',
                                                                                            x = x, p = 0.01))  

bins_factor <- lapply(names(dt_f) [ind], function(x) smbinning.factor(df = dt_f, 
                                                                      y = 'reclassified',
                                                                      x = x))
breaks_adj <- list()

for (i in 1:length(x = bins_factor)) {
  var_name <- bins_factor[[i]]$x
  append(breaks_adj, breaks_adj[[var_name]] <- bins_factor[[i]]$cuts)
}

for (i in 1:length(x = bins_num)) {
  var_name <- bins_num[[i]]$x
  append(breaks_adj, breaks_adj[[var_name]] <- bins_num[[i]]$cuts)
}

dt_list <- split_df(dt_f, y = 'reclassified', ratio = 0.7, seed = seed)

# binning
bins_adj <- lapply(dt_list, function(dat) {
  suppressWarnings(woebin(dat, y = "reclassified", x = NULL, breaks_list = breaks_adj, special_values = NULL, print_info=FALSE))
})

# converting train and test into woe values
dt_woe_list <- lapply(dt_list, function(dat) {
  suppressWarnings(woebin_ply(dat, bins_adj[[1]], print_info=FALSE))
})

# glm 

cl <- makeCluster(2)
registerDoParallel(cl)
on.exit(stopCluster(cl))

# define training control
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

# train the model on training set
m2 <-  suppressWarnings(train(factor(reclassified) ~ .,
                              data = dt_woe_list$train,
                              trControl = train_control,
                              method = "glm",
                              family= 'binomial'))
# print cv scores
summary(m2$finalModel)
# predict on train and test set
pred_list <- lapply(dt_woe_list, function(x) predict(m2, x, type = 'prob') [[2]])

# performance ks & roc 
## predicted probability

## performance
cutoff <- seq(.1,.9,0.01)
sens <- map_dbl(cutoff, function(x){
  p_hat <- pred_list$train
  y_hat <-  ifelse(p_hat > x, '1','0') %>% factor()
  sensitivity(data = factor(dt_woe_list$train$reclassified), reference =  y_hat)
})
max(spec)
best_cutoff <- cutoff[which.max(sens)]
best_cutoff

p_hat <- pred_list$test
y_hat <- ifelse(p_hat > best_cutoff, "1", "0") %>% factor()
perf <- perf_eva(pred = pred_list, label = label_list, show_plot =  c('ks', 'lift', 'gain', 'roc', 'lz',
                                                                      'pr', 'f1', 'density'), threshold = best_cutoff)
perf$binomial_metric

perf$confusion_matrix
detect_prevalence <- as.numeric(perf$confusion_matrix$train[3,2] / (perf$confusion_matrix$train[3,2] + 
                                                                  perf$confusion_matrix$train[3,3]))

confusionMatrix(y_hat,factor(dt_woe_list$test$reclassified), positive = '1')

# score ------
## scorecard
card <- scorecard(bins = bins_adj[[1]], model = m2$finalModel)

## credit score
# calculates credit score using the results from scorecard
score_list <- lapply(dt_list, function(x) scorecard_ply(x, card))
score_list_order <- score_list$test %>% arrange(desc(score))
score_cutoff <- score_list_order[detect_prevalence*dim(score_list_order) [1], 1]
score_cutoff

# includes both total and each variable's credit score.
score_list2 <- lapply(dt_list, function(x) scorecard_ply(x, card, only_total_score = FALSE))
head(score_list$train, 20) %>% knitr::kable()


# specify the bins number and type
gains_table(score = score_list, label = label_list, bin_num = 30, bin_type = 'width')

# Report


suppressWarnings(report(list(train = dt_list$train, test = dt_list$test), y = 'reclassified',
                        x = cols_not_remove, breaks_list = breaks_adj, special_values = NULL,
                        seed = seed,  save_report='report1', show_plot = c('ks', 'lift', 'gain',
                                                                           'roc', 'lz', 'pr', 'f1', 'density'),
                        bin_type = 'width'))

# Statistical Tests for Rating System Calibration
# The problem with calibration of rating systems or score variables is comparing
# the realized default frequency with the estimates of the conditional default probability
# given the score and analyzing the difference between the observed default frequency
# and the estimated probability of default. There are several statistical methods for 
# validating the probability of default, such as the Binomial test, the Spiegelhalter test
# and the Hosmer-Lemeshow Chi-square test. While the binomial test can only be applied to
# one single rating grade over a single time period,Spiegelhalter test and Hosmer-Lemeshow (X^2)
# test provide more advanced methods that can be used to test the adequacy of the default 
# probability prediction over a single time period for several rating grades.

#Hosmer - Lemeshow test
y <- dt_woe_list$test$reclassified
fit <- predict(m2$finalModel, dt_woe_list$test, type = 'response') 
g <- 10
hl_test <- hoslem.test(y, fit, g = g)
hl_test
hl_table <- cbind(hl_test$observed,round(hl_test$expected))
knitr::kable(hl_table, format = "pandoc",
             caption = "Hosmer Lemeshow",
             col.names = c('obs_good','obs_bad',
                           'pred_good', 'pred_bad'))

# Spiegelhalter test https://esc.fnwi.uva.nl/thesis/centraal/files/f1668180724.pdf
# Normally the predicted default probability of each borrower is individually calculated.
# Since the Hosmer-Lemeshow Chi-square test requires averaging the predicted PDs of customers
# that have been classified within the same rating class, some bias might arise in the 
# calculation. One could avoids this problem by using the Spiegelhalter test.


n_obs <- length(fit)
mean_square_error <- mean((y - fit)^2)                          # mean square error
E_mse <- mean(fit * (1 - fit))                                  # expected mean square error
var_mse <- (sum(fit * (1 - fit) * ((1 - 2 * fit)^2))) / n_obs^2  # variance mean square error
z_stat <- abs((mean_square_error  - E_mse ) / sqrt(var_mse))    # z statistic
print(c("Spiegelhalter test Statistic" = z_stat, "p-value" = pnorm(z_stat,lower.tail=F)))

# The result of Hosmer-Lemeshow Chi-square test and Spiegelhalter test suggest that
# according to the p-values, the estimated PD are quite close to the observed default
# rates.

# Binomial test -------
library(creditR)
obs <- hl_table[,1] + hl_table[,2]
pd_hat <- round(hl_table[ ,4] / obs, 3)
bad.rate.obs <- round(hl_table[ ,2] / obs, 3)
hl_table <- cbind(obs, hl_table, bad.rate.obs, pd_hat)

# independent default
bt_table <- Binomial.test(hl_table, "obs", "pd_hat" , "bad.rate.obs", 0.90, "one")

knitr::kable(bt_table, format = "pandoc",
             caption = "Binomial test - independient defaults",
             col.names = c('obs','good','bad',
                           'pred_good', 'pred_bad',
                           'bad_rate','PD',
                           'bad_rate',
                           'bad_expected','test_estimate',
                           'test_result'))
# correlated defaults
adj_bt_table <- Adjusted.Binomial.test(hl_table, "obs", "pd_hat" , "bad.rate.obs", 0.90, "one", r = 0.3)
knitr::kable(adj_bt_table, format = "pandoc",
             caption = "Binomial test - correlated defaults",
             col.names = c('obs','good','bad',
                           'pred_good', 'pred_bad',
                           'bad_rate','PD',
                           'calc_bad_rate',
                           'bad_exp','corr',
                           'test_estimate',
                           'test'))

# ROC and Gini test
gini <- rcorr.cens(fit, y)
ROC <- gini['C Index']
gini_test <- gini['Dxy']
print(c("R.O" = ROC))
