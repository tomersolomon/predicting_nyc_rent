## Tomer Solomon Mate (ts2838)
## HW2 - Machine Learning for NYC Apartment Rents

###Overall Task

Regression task to predict the monthly rent of an NYC apartment based on Census Data, specifically the NYC Housing and Vacancy survey.


###Feature Selection

For feature selection I manually went through data set first, starting with vacancy (so I don't count features that depend on the person living like race etc...).  It's important to split up these features into continuous and categorical features. For categorical features I then did one hot encoding to allow a regression to work.

###Imputation

One hot encoding doesn't work without imputation, so I used Imputation(most frequent). It's important to do one_hot encoding AFTER train/test/split.
Imputation needs to work for any new data.


###Model

Lasso model.

###Validation

Cross-validation with k=10.


