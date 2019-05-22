########################################################################################################
#                                        Summary
########################################################################################################
# A  bank is active in the loans with repayment through payroll deductions a employees of the state    #
# goverment. Those employees have legal job stability, so is very difficult that an employee is fired. #
# Given both, job stability and the sure payment, the default probability is very low, but argentine   #
# banking regulator requires that under certain circumstances a bank must provisioning (charge a loss) #
# to debtor paying in a timely manner the debt contracted but delay pays of debt with other bank.      #
# This become very expensive for the bank, so, the bank want avoid give a loan to whose can fall in    #
# this situation.                                                                                      #
#                                                                                                      #
# The bank requiered us to design a model that predict if a debtor will be reclassified durant the     #
# next year, The bank also requires build a score system and pick a cutoff that reach a reclassified   #
# rate lesser than 0.045, but allowing high approval rate..                                            #
#                                                                                                      #
# The data were provided by the bank. The bank only chose  those debtors whose classification differs  #
# from their, but is not yet required to be reclassified for not fulfilling some of the three          #
# conditions set forth in the regulations. In short, data do not contains debtors that paying in time  #
# and form theirs debts in all financial system argentine neither debtors reclassified in T0 (Figure 1)#
#                                                                                                      #
# The model is completed through the following steps:                                                  #
#                                                                                                      #
# 1  Automated optimal Weight-of-Evidence (WOE) and Information Value (IV) binning is executed.        #
# 2  The variates with Information Value (IV) bins with greater power are selected.                    #
# 3  Transformation all features within the data set so that the new data contains only                #
#    weight-of-evidence valued transformations.                                                        #
# 4  Models using logistic regression is created.                                                      #
# 5  Credit scorecard is created.                                                                      #
# 6  Cutoff that maximize the sensibility, that is the ability of our model to correctly identify      #
#    those debtors reclassified (true positive rate) but allowing a high the approval rate, is         #
#    computed.                                                                                         #
# 7  Good of fit statistic and Statistical measures of discriminatory power are calculated.            #
########################################################################################################

# required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(Hmisc)) install.packages("Hmisc", repos = "http://cran.us.r-project.org")
if(!require(doParallel)) install.packages("doParallel", repos = "http://cran.us.r-project.org")
if(!require(ResourceSelection)) install.packages("ResourceSelection", repos = "http://cran.us.r-project.org")
if(!require(devtools)) install.packages("devtools", repos = "http://cran.us.r-project.org")
if(!require(rlang)) install.packages("rlang", repos = "http://cran.us.r-project.org")
if(!require(smbinning)) install.packages("smbinning", repos = "http://cran.us.r-project.org")
devtools::install_github("shichenxie/scorecard", force = FALSE)

library(tidyverse)
library(caret)
library(scorecard)  # creates a scorecard

# Optimal Binning categorizes a numeric characteristic into bins 
# for ulterior usage in scoring modeling.
source_url("https://raw.githubusercontent.com/mrmelchi/Credit_Scoring/master/smbinning.R")  

# using to solve 'Error in summary.connection(connection) : invalid connection'
# https://stackoverflow.com/questions/25097729/un-register-a-doparallel-cluster
library(doParallel) 
library(smbinning)         # compute woe and IV
library(ResourceSelection) # Compute Hosmer Lemeshow test
library(Hmisc)             # Compute Gini and ROC index

# load data from github
url <- "https://raw.githubusercontent.com/mrmelchi/Credit_Scoring/master/data.csv"
data <- read.csv(url, sep=";")
# show structure and dimension of data set
str(data)
dim(data)

# change some variates's class, derive sex variate from tax_identification_number and delete tax_identification_number
data$repartition <- factor(data$repartition)

data <- data %>%
  mutate(reclassified = ifelse(reclassified == 'yes',1L,0L)) %>%
  mutate(tax_identification_number = floor(tax_identification_number / 10^9)) %>%
  mutate(sex = factor(ifelse(tax_identification_number == 27 | tax_identification_number == 23,
                              'Female', 'Male'))) %>%
  mutate(reclassified_prior_11_months = factor(ifelse(reclassified_prior_11_months == '0','no','yes'))) %>% 
  select(-tax_identification_number)

# compute automatic optimal bins
opt_bins <- smbinning.sumiv(data, y = 'reclassified')
opt_bins %>% knitr::kable()

# erase variates with No significant splits IV
no_sig <- opt_bins %>% filter(is.na(IV))
data <- data %>%
  select(-c( as.character(no_sig$Char)))

# compute automatic optimal bins from rest of variates
opt_bins <- smbinning.sumiv(data, y = 'reclassified')
opt_bins %>% knitr::kable()
smbinning.sumiv.plot(opt_bins, cex = 0.9)

# select variates with IV >= 0.02
data_opt <- data %>% 
  select(as.character(opt_bins$Char [opt_bins$IV >= 0.02]), reclassified)

# find out the class each variate
class <- lapply(data_opt, class)
# index the factor variate
ind <- which(class == 'factor')

# create bins from numeric variate. 
# the code coerce monotonicity
bins_num <- lapply(names(data_opt) [-c(ind,length(class))], function(x) smbinning.monotonic(df = data_opt, 
                                                                                  y = 'reclassified',
                                                                                  x = x, p = 0.01)) 

# create bins from factor variate.
bins_factor <- lapply(names(data_opt) [ind], function(x) smbinning.factor(df = data_opt, 
                                                                          y = 'reclassified',
                                                                          x = x)) 
# Traditional Credit Scoring Using Logistic Regression
# This is a randomized algorithm
seed <- 123

# save the variates selected
keep <- names(data_opt) [-length(data_opt)]

# filter variables
dt_f <- var_filter(data_opt, y = 'reclassified', positive = 1, var_kp = keep)

# split a dataset into train and test
dt_list <- split_df(dt_f, y = 'reclassified', ratio = 0.7, seed = seed)
# create list from label class for train and test
label_list <- lapply(dt_list, function(x) x$reclassified)

# use bins create from smbinning package
breaks_adj <- list()

for (i in 1:length(x = bins_factor)) {
  var_name <- bins_factor[[i]]$x
  append(breaks_adj, breaks_adj[[var_name]] <- bins_factor[[i]]$cuts)
}

for (i in 1:length(x = bins_num)) {
  var_name <- bins_num[[i]]$x
  append(breaks_adj, breaks_adj[[var_name]] <- bins_num[[i]]$cuts)
}

# adjust binning breaks as scorecard package needs 
bins_adj <- suppressWarnings(woebin(dt_f, y = "reclassified", breaks_list = breaks_adj, 
                                    var_kp = keep, save_breaks_list = NULL))

# transform train and test into woe values
dt_woe_list = lapply(dt_list, function(x) suppressWarnings(woebin_ply(x, bins_adj)))


# Solve 'Error in summary.connection(connection) : invalid connection'
# https://stackoverflow.com/questions/25097729/un-register-a-doparallel-cluster
cl <- makeCluster(2)
registerDoParallel(cl)
on.exit(stopCluster(cl))

# variables selection by Regularization
# define training control
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

# select variates using cross validation and glmnet
set.seed(seed)
m1 <- train(factor(reclassified) ~ .,
            data = dt_woe_list$train,
            trControl = train_control,
            method = "glmnet",
            family= 'binomial')

# remove variates with coefficient equal to zero
m1_reg <- coef(m1$finalModel, m1$bestTune$lambda) [-1,1] != 0
cols_not_remove <-  str_remove(c(names(coef(m1$finalModel, m1$bestTune$lambda) [-1,1])),'_woe') [m1_reg]
y <- dt_f$reclassified
cols <- names(dt_f) 
dt_f <- dt_f %>% select(match(cols_not_remove, cols))
dt_f <- dt_f %>% mutate(reclassified = y)

# find out the class each variate
class <- lapply(dt_f, class)
# index the factor variate
ind <- which(class == 'factor')

# create bins from numeric variates remaining
# the code coerce monotonicity
bins_num <- lapply(names(dt_f) [-c(ind,length(class))], function(x) smbinning.monotonic(df = dt_f, 
                                                                                            y = 'reclassified',
                                                                                            x = x, p = 0.01))  
# create bins from factor variates remaining
bins_factor <- lapply(names(dt_f) [ind], function(x) smbinning.factor(df = dt_f, 
                                                                      y = 'reclassified',
                                                                      x = x))

# use bins create from smbinning package of de variates remaning from regularization
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

# adjust binning breaks of remaining variates from regularization as scorecard package needs
bins_adj <- lapply(dt_list, function(dat) {
  suppressWarnings(woebin(dat, y = "reclassified", x = NULL, breaks_list = breaks_adj, special_values = NULL, print_info=FALSE))
})

# converting train and test into woe values
dt_woe_list <- lapply(dt_list, function(dat) {
  suppressWarnings(woebin_ply(dat, bins_adj[[1]], print_info=FALSE))
})

# Solve 'Error in summary.connection(connection) : invalid connection'
# https://stackoverflow.com/questions/25097729/un-register-a-doparallel-cluster
cl <- makeCluster(2)
registerDoParallel(cl)
on.exit(stopCluster(cl))

# Design Logistic regression. scorecard package needs a glm object
# define training control
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

# train the model on training set
m2 <-  suppressWarnings(train(factor(reclassified) ~ .,
                              data = dt_woe_list$train,
                              trControl = train_control,
                              method = "glm",
                              family= 'binomial'))
# print cross validation scores
summary(m2$finalModel)
# predict on train and test set
pred_list <- lapply(dt_woe_list, function(x) predict(m2, x, type = 'prob') [[2]])

## compute cutoff for a reclassified rate lesser than 0.04 on train set
cutoff <- seq(0.05,0.15,0.001)
reclassified_rate <- map_dbl(cutoff, function(x){
  suppressMessages(cm <- perf_eva(pred = pred_list, label = label_list, show_plot =  NULL, threshold = x)$confusion_matrix)  
  as.numeric(cm$train[2,2] / cm$train[3,2])
})
# optimal cutoff
best_cutoff <- cutoff[max(which(reclassified_rate < 0.045))]
best_cutoff


# performance of the model cutoff equal to best_cutoff
perf <- perf_eva(pred = pred_list, label = label_list, show_plot =  c('ks', 'lift', 'gain', 'roc', 'lz',
                                                                      'pr', 'f1', 'density'), threshold = best_cutoff)

# measures of discriminatory power cutoff equal to best_cutoff
perf$binomial_metric
# confusion matrix cutoff equal to best_cutoff
perf$confusion_matrix

# compute reclassified rate on train and test set
reclassified_rate_train <- as.numeric(perf$confusion_matrix$train[2,2] / perf$confusion_matrix$train[3,2])
reclassified_rate_train

reclassified_rate_test <- as.numeric(perf$confusion_matrix$test[2,2] / perf$confusion_matrix$test[3,2])
reclassified_rate_test

# compute approval rate on train and test set
approval_rate_train <- as.numeric(perf$confusion_matrix$train[3,2] / (perf$confusion_matrix$train[3,2] + 
                                                                  perf$confusion_matrix$train[3,3]))
approval_rate_train

approval_rate_test <- as.numeric(perf$confusion_matrix$test[3,2] / (perf$confusion_matrix$test[3,2] + 
                                                                  perf$confusion_matrix$test[3,3]))
approval_rate_test

## scorecard
card <- scorecard(bins = bins_adj[[1]], model = m2$finalModel)

## credit score
# calculates credit score using the results from scorecard
score_list <- lapply(dt_list, function(x) scorecard_ply(x, card))

# pick score that equal to approval rate on train and test set
score_list_order_train <- score_list$train %>% arrange(desc(score))
score_cutoff_train <- score_list_order_train[approval_rate_train*dim(score_list_order_train) [1], 1]
score_cutoff_train

score_list_order_test <- score_list$test %>% arrange(desc(score))
score_cutoff_test <- score_list_order_test[approval_rate_test*dim(score_list_order_test) [1], 1]
score_cutoff_test


# compute both total and each variable's score.
score_list2 <- lapply(dt_list, function(x) scorecard_ply(x, card, only_total_score = FALSE))
# points and score each debtor
head(score_list2$train, 5)

# Gain table
gains_table(score = score_list, label = label_list, bin_num = 10, bin_type = 'width')

# generate a Report in spreadsheet 
suppressWarnings(report(list(train = dt_list$train, test = dt_list$test), y = 'reclassified',
                        x = cols_not_remove, breaks_list = breaks_adj, special_values = NULL,
                        seed = seed,  save_report='report1', show_plot = c('ks', 'lift', 'gain',
                                                                           'roc', 'lz', 'pr', 'f1', 'density'),
                        bin_type = 'width'))

# goodness-of-fit test
# Hosmer - Lemeshow test
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
print(c("Spiegelhalter test Statistic" = z_stat, "p-value" = pnorm(z_stat,lower.tail=T)))

# The result of Hosmer-Lemeshow Chi-square test and Spiegelhalter test suggest that
# according to the p-values, the estimated PD are quite close to the observed default
# rates.

