import joblib
import numpy as np

forest_reg = joblib.load('models/random_forest.pkl')
num_imputer = joblib.load('models/num_imputer.pkl')
num_scaler = joblib.load('models/num_scaler.pkl')
cat_imputer = joblib.load('models/cat_imputer.pkl')
cat_encoder = joblib.load('models/cat_encoder.pkl')

# load the test data
import pandas as pd

test = pd.read_csv('data/test.csv')

# separate features and target
X_test = test.drop('median_house_value', axis=1)
y_test = test['median_house_value'].copy()

# separate numerical and categorical features
num_features = X_test.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X_test.select_dtypes('object').columns.tolist()

X_test_num = X_test[num_features]
X_test_cat = X_test[cat_features]

# apply transformations
X_test_num = num_imputer.transform(X_test_num)
X_test_num = num_scaler.transform(X_test_num)

X_test_cat = cat_imputer.transform(X_test_cat)
X_test_cat = cat_encoder.transform(X_test_cat)

# combine numerical and categorical features
X_test = np.concatenate([X_test_num, X_test_cat], axis=1)

# make predictions
y_pred = forest_reg.predict(X_test)

# evaluate the model
from sklearn.metrics import root_mean_squared_error

rmse = np.sqrt(root_mean_squared_error(y_test, y_pred))

print(f'RMSE: {rmse}')
