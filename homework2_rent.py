from __future__ import division, absolute_import, print_function

import numpy as np
import scipy as sp

from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV #model_selection only works in python3
from sklearn.preprocessing import Imputer,OneHotEncoder,StandardScaler, MaxAbsScaler, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

import pandas as pd

def score_rent():
	"""Function that applies Regression with Lasso Regularization to the NYC Census data to predict the monthly rent of an partment.


	Returns
	-------
	int
		R^2 value. 

	"""	

	url = 'https://ndownloader.figshare.com/files/7586326'
	raw_data = pd.read_csv(url)

	#delete 9999 which is missing value (can't train if there is no response variable)
	raw_data = raw_data[raw_data.uf17 <= 9999]

	##MANUAL FEATURE SELECTION

	# Categorical features (sorted by types of NaN)

	#no NaN values
	cat_features_none = ['boro','cd','new_csr','sc26','sc149','sc152','sc153','sc155','sc156','sc158']

	# 8 is Nan
	cat_features_8 = ['uf1_1','uf1_2','uf1_3','uf1_4','uf1_5','uf1_6','uf1_7','uf1_8','uf1_9','uf1_10','uf1_11','uf1_12','uf1_13','uf1_14','uf1_15','uf1_16','uf1_35','uf1_17','uf1_18','uf1_19','uf1_20','uf1_21','uf1_22','sc23','sc24','sc36','sc37','sc38','sc147','sc171','sc154','sc157','sc197','sc198','sc188','sc190','sc191','sc192','sc193','sc194','sc575']

	#4 is nan
	cat_features_4 = ['sc114']

	#added up all cat features
	total_cat = cat_features_none + cat_features_8 + cat_features_4


	#Continous features (sorted by types of NaN)

	cont_features_none = ['sc150','sc151','fw']

	#98,99 are NaN
	cont_features_99 = ['uf11','uf23']

	#9999 is NaN (mostly monthly fees)

	cont_features_9999 = ['uf12','uf13','uf14','uf15','sc186']

	# 8 is NaN
	cont_features_8 = ['sc189','sc196','sc199']

	total_cont = cont_features_none + cont_features_99 + cont_features_9999 + cont_features_8  


	## Choose the specified categorical and continuous features

	X = raw_data[total_cat + total_cont]

	## Replace all missing values to NaN's

	# Continuous feature replacement
	X[cont_features_99] = X[cont_features_99].replace([99,98],np.nan)
	X[cont_features_9999] = X[cont_features_9999].replace([9999,99999],np.nan)
	X[cont_features_8 ] = X[cont_features_8 ].replace(8,np.nan)


	# Categorical feature replacement
	X[cat_features_8] = X[cat_features_8].replace(8,np.nan)
	X[cat_features_4] = X[cat_features_4].replace(8,np.nan)


	# Turn categorical features into dummies 
	for i in total_cat:
		X[i] = X[i].astype('category')

	#TRAIN TEST SPLIT

	#train test split
	y=raw_data['uf17']

	X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)

	#This performs the one hot encoding
	
	X_train = pd.get_dummies(X_train) 
	
	X_test = pd.get_dummies(X_test) 


	#aglorithmic feature selection
	select_ridgecv = SelectFromModel(RidgeCV(),threshold = 'median')

	#create of pipeline
	pipe = make_pipeline(Imputer(missing_values='NaN',strategy='most_frequent'),MaxAbsScaler(),select_ridgecv,Ridge(alpha=10.0))
	
	#cross validation score
	scores = cross_val_score(pipe,X_train,y_train,cv=5)

	print("scores for 5 fold cv:")
	print("cross val scores:")
	print(scores)

	model = pipe.fit(X_train,y_train)
	predicted_label = model.predict(X_test)
	print("r^2 value is:")
	print (r2_score(y_test, predicted_label))

	#gridsearchCV to determine proper alpha to use for ridge regression
	#lower value?

	def grid_search():
		print('gridsearch')
		param_grid = {'ridge__alpha' : np.array([0.001,0.01,.1,1,10,100])}
		grid = GridSearchCV(pipe,param_grid,cv=5)
		grid.fit(X_train,y_train)
		#grid.predict(X_test)
		print("grid search")
		print(grid.score(X_test,y_test))
		print(grid.best_params_)

	return r2_score(y_test, predicted_label)

score_rent()

def predict_rent():
	"""Function that also applies Regression with Lasso Regularization to the NYC Census data to predict the monthly rent of an partment.


	Returns
	-------
	numpy.array
		Test Data 
	numpy.array
		True Labels
	numpy.array
		Predicted Labels
		
	"""

	url = 'https://ndownloader.figshare.com/files/7586326'
	raw_data = pd.read_csv(url)

	#delete 9999 which is missing value (can't train if there is no response variable)
	raw_data = raw_data[raw_data.uf17 <= 9999]

	##MANUAL FEATURE SELECTION

	# Categorical features (sorted by types of NaN)

	#no NaN values
	cat_features_none = ['boro','cd','new_csr','sc26','sc149','sc152','sc153','sc155','sc156','sc158']

	# 8 is Nan
	cat_features_8 = ['uf1_1','uf1_2','uf1_3','uf1_4','uf1_5','uf1_6','uf1_7','uf1_8','uf1_9','uf1_10','uf1_11','uf1_12','uf1_13','uf1_14','uf1_15','uf1_16','uf1_35','uf1_17','uf1_18','uf1_19','uf1_20','uf1_21','uf1_22','sc23','sc24','sc36','sc37','sc38','sc147','sc171','sc154','sc157','sc197','sc198','sc188','sc190','sc191','sc192','sc193','sc194','sc575']

	#4 is nan
	cat_features_4 = ['sc114']

	#added up all cat features
	total_cat = cat_features_none + cat_features_8 + cat_features_4


	#Continous features (sorted by types of NaN)

	cont_features_none = ['sc150','sc151','fw']

	#98,99 are NaN
	cont_features_99 = ['uf11','uf23']

	#9999 is NaN (mostly monthly fees)

	cont_features_9999 = ['uf12','uf13','uf14','uf15','sc186']

	# 8 is NaN
	cont_features_8 = ['sc189','sc196','sc199']

	total_cont = cont_features_none + cont_features_99 + cont_features_9999 + cont_features_8  


	## Choose the specified categorical and continuous features

	X = raw_data[total_cat + total_cont]

	## Replace all missing values to NaN's

	# Continuous feature replacement
	X[cont_features_99] = X[cont_features_99].replace([99,98],np.nan)
	X[cont_features_9999] = X[cont_features_9999].replace([9999,99999],np.nan)
	X[cont_features_8 ] = X[cont_features_8 ].replace(8,np.nan)


	# Categorical feature replacement
	X[cat_features_8] = X[cat_features_8].replace(8,np.nan)
	X[cat_features_4] = X[cat_features_4].replace(8,np.nan)


	# Turn categorical features into dummies 
	for i in total_cat:
		X[i] = X[i].astype('category')


	#TRAIN TEST SPLIT

	#train test split
	y=raw_data['uf17']

	X_train_preOHE,X_test,y_train,y_test = train_test_split(X,y,random_state=1)


	#imp needs to be before OHE
	#imp = Imputer(missing_values='NaN',strategy='most_frequent')
	#X_train = imp.fit_transform(X_train)
	#X_test = imp.fit_transform(X_test)

	#This performs the one hot encoding
	X_train = pd.get_dummies(X_train_preOHE) 
	X_test = pd.get_dummies(X_test) 


	#aglorithmic feature selection
	select_ridgecv = SelectFromModel(RidgeCV(),threshold = 'median')

	#create of pipeline
	pipe = make_pipeline(Imputer(missing_values='NaN',strategy='most_frequent'),MaxAbsScaler(),select_ridgecv,Ridge(alpha=10.0))
	
	#cross validation score
	#scores = cross_val_score(pipe,X_train,y_train,cv=5)

	#print("scores for 5 fold cv:")
	#print("cross val scores:")
	#print(scores)

	model = pipe.fit(X_train,y_train)
	predicted_label = model.predict(X_test)

	# everything above is same code as score_rent except for making X_train,X_test preOHE varaibles
	# print('r2 value')
	# print (r2_score(y_test, predicted_label))

	#create numpy arrays out of X_train and y_test
	
	print("X_train,y_test, and predicted_label are:")
	
	#want to return preOHE since we want to do before OHE has been applied as well as imputing

	print(X_train_preOHE.as_matrix(),y_test.as_matrix(),predicted_label)
	return(X_train_preOHE.as_matrix(),y_test.as_matrix(),predicted_label)

predict_rent()













