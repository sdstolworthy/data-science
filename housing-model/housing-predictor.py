#!/usr/bin/python3
import operator
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)


MELBOURNE_FILE_PATH = './data/train.csv'
unprocessed_data = pd.read_csv(MELBOURNE_FILE_PATH)
melbourne_data = unprocessed_data.copy()
y = melbourne_data.SalePrice
PREDICTOR_COLUMNS = ['LotArea', 'YearBuilt', '1stFlrSF',
                     '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = melbourne_data[PREDICTOR_COLUMNS]

cols_with_missing = (col for col in X.columns if X[col].isnull().any())

for col in cols_with_missing:
  print(col)
  X[col+'_was_missing'] = X[col].isnull()

imputer = Imputer()
X = imputer.fit_transform(X)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
mae_list = []
# for max_leaf_nodes in [5, 50, 500, 5000]:
#     mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
#     mae_list.append((max_leaf_nodes, mae))
#     print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %
#           (max_leaf_nodes, mae))
forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
forest_predictions = forest_model.predict(val_X)
TEST_FILE = './data/test.csv'
test = pd.read_csv(TEST_FILE)
test_X = test[PREDICTOR_COLUMNS]
predicted_prices = forest_model.predict(test_X)
print(predicted_prices)
submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
submission.to_csv('submission.csv', index=False)
# min_index, min_value = min(
#     enumerate(x[1] for x in mae_list), key=operator.itemgetter(1))
# print('min', mae_list[min_index])