#!/usr/bin/python

import pandas as pd

house_data = pd.read_csv('./data/train.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds_val = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds_val)
    return(mae)

def data_with_dropped_columns(X_train, X_test, y_train, y_test):
    cols_with_missing = [
        col for col in X_train.columns if X_train[col].isnull().any()]
    reduced_X_train = X_train.drop(cols_with_missing, axis=1)
    reduced_X_test = X_test.drop(cols_with_missing, axis=1)
    print("MAE dropped cols:")
    print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))

def data_with_imputation(X_train, X_test, y_train, y_test):
    imputer = Imputer()
    imputed_X_train = imputer.fit_transform(X_train)
    imputed_X_test = imputer.transform(X_test)
    print("MAE Imputation:")
    print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))

def data_with_imputation_extra_columns(X_train, X_test, y_train, y_test):
    imputed_X_train_plus = X_train.copy()
    imputed_X_test_plus = X_test.copy()
    cols_with_missing = (col for col in X_train.columns if X_train[col].isnull().any())    
    for col in cols_with_missing:
        imputed_X_train_plus[col+'_was_missing'] = imputed_X_train_plus[col].isnull()
        imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
    
    imputer = Imputer()
    imputed_X_train_plus = imputer.fit_transform(imputed_X_train_plus)
    imputed_X_test_plus = imputer.transform(imputed_X_test_plus)
    print("MAE imputed w/ columns:")
    print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))

target_value = house_data.SalePrice
predictors = house_data.drop(['SalePrice'], axis=1)
numeric_predictors = predictors.select_dtypes(exclude=['object'])
X_train, X_test, y_train, y_test = train_test_split(numeric_predictors, target_value, random_state = 0)

data_with_dropped_columns(X_train,X_test, y_train, y_test)
data_with_imputation(X_train,X_test, y_train, y_test)
data_with_imputation_extra_columns(X_train,X_test, y_train, y_test)

