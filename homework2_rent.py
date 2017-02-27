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
	url = 'https://ndownloader.figshare.com/files/7586326'
	raw_data = pd.read_csv(url)

	#delete 9999 which is missing value (can't train if there is no response variable)
	raw_data = raw_data[raw_data.uf17 <= 9999]

	##MANUAL FEATURE SELECTION

	# Categorical features

	#none are NaN
	cat_features = ['boro','cd']

	# 8 is nan
	cat_featured_added = ['uf1_1','uf1_2','uf1_3','uf1_4','uf1_5','uf1_6','uf1_7','uf1_8','uf1_9','uf1_10']
	cat_featured_test_10 = ['uf1_11','uf1_12','uf1_13','uf1_14','uf1_15','uf1_16','uf1_35','uf1_17','uf1_18']
	cat_featured_test_11 =['uf1_19','uf1_20','uf1_21','uf1_22','sc23','sc24','sc36','sc37','sc38']
	cat_featured_test_12 = ['sc147','sc171','sc154','sc157','sc197','sc198','sc188','sc190','sc191','sc192','sc193','sc194','sc575'] 

	#cat_simple = ['uf1_1','uf1_7','uf1_7','uf1_17']
	#cat_simple1 = ['sc23','sc24','sc36','sc37','sc38']

	#4 is nan
	cat_featured_added_1 = ['sc114']

	#no nan values
	cat_featured_added_2 = ['sc149','sc152','sc153','sc155','sc156','sc158']

	total_cat = cat_features + cat_featured_added_1 + cat_featured_added + cat_featured_test_10 + cat_featured_test_11 + cat_featured_test_12 

	#total_cat_simple = cat_simple + cat_simple1 + cat_featured_test_12 + cat_featured_added_1 + cat_featured_added_2

	#Continous

	cont_features = ['sc150','sc151','fw']

	#98,99 nan, 13k of those
	#monthly fees, real estate taxes (13.7K nan), stories in building
	cont_added = ['uf11','uf23']

	#9999 is nan (mostly monthly fees)
	#sc186 9 is no breakdowns, change to 0

	cont_added_1 = ['uf12','uf13','uf14','uf15','sc186']

	# 8 is not reported
	cont_added_2 = ['sc189','sc196','sc199']

	#differences
	#sc143,sc144,sc154

	total_cont = cont_features + cont_added + cont_added_1 + cont_added_2 


	## REPLACE INTEGERS WITH NaN's

	X = raw_data[total_cat + total_cont]

	#cont
	X[cont_added] = X[cont_added].replace([99,98],np.nan)
	X[cont_added_1] = X[cont_added_1].replace([9999,99999],np.nan)
	X[cont_added_2 ] = X[cont_added_2 ].replace(8,np.nan)


	#cat
	X[cat_featured_added] = X[cat_featured_added].replace(8,np.nan)
	X[cat_featured_test_10] = X[cat_featured_test_10].replace(8,np.nan)
	X[cat_featured_test_11] = X[cat_featured_test_11].replace(8,np.nan)
	X[cat_featured_test_12] = X[cat_featured_test_12].replace(8,np.nan)
	X[cat_featured_added_1] = X[cat_featured_added_1].replace(8,np.nan)


	## ONE HOT ENCODING warmup (imputation needs to happen before this)

	#turn categorical data into dummies
	for i in total_cat:
		X[i] = X[i].astype('category')


	#TRAIN TEST SPLIT

	#train test split
	y=raw_data['uf17']

	X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)


	#imp needs to be before OHE
	#imp = Imputer(missing_values='NaN',strategy='most_frequent')
	#X_train = imp.fit_transform(X_train)
	#X_test = imp.fit_transform(X_test)

	#do the OHE
	X_train = pd.get_dummies(X_train) 
	X_test = pd.get_dummies(X_test) 


	#feature selection
	select_ridgecv = SelectFromModel(RidgeCV(),threshold = 'median')

	#pipeline

	pipe = make_pipeline(Imputer(missing_values='NaN',strategy='most_frequent'),MaxAbsScaler(),select_ridgecv,Ridge(alpha=10.0))
	scores = cross_val_score(pipe,X_train,y_train,cv=5)

	#print("scores for 5 fold cv:")
	print("cross val scores:")
	print(scores)

	model = pipe.fit(X_train,y_train)
	predicted_label = model.predict(X_test)
	print("r^2 value is:")
	print (r2_score(y_test, predicted_label))

	#gridsearchCV
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

	grid_search()

	#IMPUTATION (REPLACING NAN'S)

	# imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
	# imp.fit_transform(X_train)

	# # #LASSO MODEL

	# lasso = Lasso(alpha=0.001).fit(X_train,y_train)
	# print(lasso.score(X_train,y_train))
	# print (lasso.score(X_test,y_test))
	# print(np.sum(lasso.coef_ != 0))

	#print (X_dummies.head())

	return r2_score(y_test, predicted_label)


score_rent()

def predict_rent():

	url = 'https://ndownloader.figshare.com/files/7586326'
	raw_data = pd.read_csv(url)

	#delete 9999 which is missing value (can't train if there is no response variable)
	raw_data = raw_data[raw_data.uf17 <= 99999]

	##MANUAL FEATURE SELECTION

	# Categorical features

	#none are NaN
	cat_features = ['boro','cd']

	# 8 is nan
	cat_featured_added = ['uf1_1','uf1_2','uf1_3','uf1_4','uf1_5','uf1_6','uf1_7','uf1_8','uf1_9','uf1_10']
	cat_featured_test_10 = ['uf1_11','uf1_12','uf1_13','uf1_14','uf1_15','uf1_16','uf1_35','uf1_17','uf1_18']
	cat_featured_test_11 =['uf1_19','uf1_20','uf1_21','uf1_22','sc23','sc24','sc36','sc37','sc38']
	cat_featured_test_12 = ['sc147','sc171','sc154','sc157','sc197','sc198','sc188','sc190','sc191','sc192','sc193','sc194','sc575'] 

	#cat_simple = ['uf1_1','uf1_7','uf1_7','uf1_17']
	#cat_simple1 = ['sc23','sc24','sc36','sc37','sc38']

	#4 is nan
	cat_featured_added_1 = ['sc114']

	#no nan values
	cat_featured_added_2 = ['sc149','sc152','sc153','sc155','sc156','sc158']

	total_cat = cat_features + cat_featured_added_1 + cat_featured_added + cat_featured_test_10 + cat_featured_test_11 + cat_featured_test_12 

	#total_cat_simple = cat_simple + cat_simple1 + cat_featured_test_12 + cat_featured_added_1 + cat_featured_added_2

	#Continous

	cont_features = ['sc150','sc151']

	#98,99 nan, 13k of those
	#monthly fees, real estate taxes (13.7K nan), stories in building
	cont_added = ['uf11','uf23']

	#9999 is nan (mostly monthly fees)
	#sc186 9 is no breakdowns, change to 0

	cont_added_1 = ['uf12','uf13','uf14','uf15','sc186']

	# 8 is not reported
	cont_added_2 = ['sc189','sc196','sc199']

	#differences
	#sc143,sc144,sc154

	total_cont = cont_features + cont_added + cont_added_1 + cont_added_2 


	## REPLACE INTEGERS WITH NaN's

	X = raw_data[total_cat + total_cont]

	#cont
	X[cont_added] = X[cont_added].replace([99,98],np.nan)
	X[cont_added_1] = X[cont_added_1].replace([9999,99999],np.nan)
	X[cont_added_2 ] = X[cont_added_2 ].replace(8,np.nan)


	#cat
	X[cat_featured_added] = X[cat_featured_added].replace(8,np.nan)
	X[cat_featured_test_10] = X[cat_featured_test_10].replace(8,np.nan)
	X[cat_featured_test_11] = X[cat_featured_test_11].replace(8,np.nan)
	X[cat_featured_test_12] = X[cat_featured_test_12].replace(8,np.nan)
	X[cat_featured_added_1] = X[cat_featured_added_1].replace(8,np.nan)


	## ONE HOT ENCODING warmup (imputation needs to happen before this)

	#turn categorical data into dummies
	for i in total_cat:
		X[i] = X[i].astype('category')

	# #TRAIN TEST SPLIT

	#train test split
	y=raw_data['uf17']

	X_train_preOHE,X_test_preOHE,y_train,y_test = train_test_split(X,y,random_state=1)


	#imp needs to be before OHE
	#imp = Imputer(missing_values='NaN',strategy='most_frequent')
	#X_train = imp.fit_transform(X_train)
	#X_test = imp.fit_transform(X_test)

	#do the OHE
	#X_train = pd.get_dummies(X_train_preOHE) 
	#X_test = pd.get_dummies(X_test_preOHE) 

	#without OHE

	X_train = X_train_preOHE 
	X_test = X_test_preOHE


	#feature selection

	select_lassocv = SelectFromModel(LassoCV(),threshold = 'mean')

	#pipeline

	pipe = make_pipeline(Imputer(missing_values='NaN',strategy='most_frequent'),MaxAbsScaler(),select_lassocv,Lasso(alpha=10))

	model = pipe.fit(X_train,y_train)
	predicted_label = model.predict(X_test)


	#everything above is same code as score_rent except for making X_train,X_test preOHE varaibles

	print(X_train_preOHE,y_test,predicted_label)
	return(X_train_preOHE,y_test,predicted_label)


#predict_rent()













