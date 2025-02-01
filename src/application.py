import pandas as pd
import numpy as np

# load data
data = pd.read_csv('data/housing.csv')

print(data.head())
print(f'Number of rows: {data.shape[0]}')
print(f'Number of columns: {data.shape[1]}')

data['income_cat'] = pd.cut(data['median_income'], 
                            bins=[0., 1.5, 3.0, 4.5, 6., np.inf], 
                            labels=[1, 2, 3, 4, 5])

# test train split
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data, 
                                       test_size=0.2, 
                                       random_state=42, 
                                       stratify=data['income_cat'])
print(f'train_set shape: {train_set.shape}')
print(f'test_set shape: {test_set.shape}')

# drop income_cat
train_set.drop('income_cat', axis=1, inplace=True)
test_set.drop('income_cat', axis=1, inplace=True)

# save train and test set
import os
os.makedirs('data', exist_ok=True)

train_set.to_csv('data/train.csv', index=False)
test_set.to_csv('data/test.csv', index=False)

print('Data saved successfully!')

# split features and target
X_train = train_set.drop('median_house_value', axis=1)
y_train = train_set['median_house_value'].copy()

# create a validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                  test_size=0.2, 
                                                  random_state=42)

print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
print(f'X_val shape: {X_val.shape}, y_val shape: {y_val.shape}')

num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X_train.select_dtypes('object').columns.tolist()

print(f'Numerical features: {num_features}')
print(f'Categorical features: {cat_features}')

# apply transformations -> numerical -> impute missing values -> scale
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

num_imputer = SimpleImputer(strategy='mean')
num_scaler = StandardScaler()

X_train_num = X_train[num_features]
X_val_num = X_val[num_features]

X_train_num = num_imputer.fit_transform(X_train_num)
X_val_num = num_imputer.transform(X_val_num)

X_train_num = num_scaler.fit_transform(X_train_num)
X_val_num = num_scaler.transform(X_val_num)

# apply transformations -> categorical -> impute missing values -> encode
cat_imputer = SimpleImputer(strategy='most_frequent')
cat_encoder = OrdinalEncoder()

X_train_cat = X_train[cat_features]
X_val_cat = X_val[cat_features]

X_train_cat = cat_imputer.fit_transform(X_train_cat)
X_val_cat = cat_imputer.transform(X_val_cat)

X_train_cat = cat_encoder.fit_transform(X_train_cat)
X_val_cat = cat_encoder.transform(X_val_cat)

# combine numerical and categorical features
X_train = np.concatenate([X_train_num, X_train_cat], axis=1)
X_val = np.concatenate([X_val_num, X_val_cat], axis=1)

print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
print(f'X_val shape: {X_val.shape}, y_val shape: {y_val.shape}')

# train a model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# evaluate the model
from sklearn.metrics import root_mean_squared_error

y_pred = lin_reg.predict(X_val)
rmse = root_mean_squared_error(y_val, y_pred)

print(f'RMSE: {rmse}')

# create a tree model
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)

# evaluate the model
y_pred = tree_reg.predict(X_val)

rmse = root_mean_squared_error(y_val, y_pred)

print(f'Tree RMSE: {rmse}')

# create random forest model
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)

# evaluate the model

y_pred = forest_reg.predict(X_val)
rmse = root_mean_squared_error(y_val, y_pred)

print(f'Random Forest RMSE: {rmse}')

# save the model
import joblib

os.makedirs('models', exist_ok=True)

joblib.dump(forest_reg, 'models/random_forest.pkl')
joblib.dump(num_imputer, 'models/num_imputer.pkl')
joblib.dump(num_scaler, 'models/num_scaler.pkl')
joblib.dump(cat_imputer, 'models/cat_imputer.pkl')
joblib.dump(cat_encoder, 'models/cat_encoder.pkl')
