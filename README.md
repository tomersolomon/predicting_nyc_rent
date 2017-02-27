## Tomer Solomon Mate (ts2838)
## HW2 - Machine Learning for NYC Apartment Rents

###Overall Task

Regression task to predict the monthly rent of an NYC apartment based on Census Data, specifically the NYC Housing and Vacancy survey.


###Feature Selection

Feature selection was definitely one of the more time consuming process. I began by first manually scraping the dataset and going through all 197 features to determine those that were relavant. For example, I discarded all the ones that were associated with the current homeowner living in the apartment, as those variables don't drive the monthly rent. I then went and sorted the relevant variables by continuous (ex) number of bedrooms, number of stories) and categorical (ex) borough, whether there are mice). This is because you can't run a regression on categorical features.

###Imputation
what I did is then applied One Hot Encoding to split the categorical features up into n binary columns, where n is the number of unique values of the feature. After this transformation,a regression can be run. I used Imputer to run an imputation with the most frequent strategy, to fill in values for the NaN's. The NaN's were time consuming to determine as well because each feature had different values for "not reported", so those had to be manually sorted. 


###Pre-Processing

I then applied a scalar, particularly the MaxsAbsScalar(), as the data was relatively sparse due to the One Hot Encoding algorithm. I also did an algorithmic feature selection using LassoCV to further narrow down the features, specifically by determining the features that were more correlated and not weighting them. To see how my model was doing, I then ran cross-validation with 5 folds.

###Model
I then trained the model using linear regression with an L2 regularization (Ridge), where the alpha determines just how much one point in the dataset influences your boundary line. Using Grid Search CV, I iterated through alpha values to determine that the best one for the model is 10.

###Results

I managed to get down to an R^2 of .326. I thought this seemed low initially when reading book examples of high correlation values, but after thinking about it, it means that a linear model might not be the best fit for the job, as well as the fact that we had alot of categorical variables that we "forced" into continuous, so a regression wouldn't fully capture the nature of these features. Also, I wonder how the extent of the imputation affects this result.

###Reflection

Overall, I thought this was a very interested assignment as we got to use real data to do prediction on NYC apartment rentals (something I definitely need to familiarize myslef with before I graduate). I learned how, contrary to what I originally thought, creating the model is the easy step; the difficult part lies in understanding and scraping the data in a way that makes sense.


###Travis-CI Link

Here is link to [Travis-CI](https://travis-ci.com/AppliedMachineLearning/homework-ii-tomersolomon) builds.




