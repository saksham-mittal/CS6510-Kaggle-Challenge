import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

training_set = pd.read_csv("train.csv")

# Extracting labels from training set
training_labels = training_set['pricing_category']
# print(training_labels)

# Dropping the last column and id from training set
training_set = training_set.drop(labels='pricing_category', axis=1)
training_set = training_set.drop(labels='id', axis=1)
# print(training_set)

# Filling nan taxi_types with new class 'O'
training_set['taxi_type'].fillna('O', inplace=True)

# Filling nan customer_scores with mean of the attribute
training_set['customer_score'].fillna(training_set['customer_score'].mean(), inplace=True)

# Filling nan customer_score_confidence with new class 'O'
training_set['customer_score_confidence'].fillna('O', inplace=True)

# Filling nan months_of_activity with 0
training_set['months_of_activity'].fillna(0.0, inplace=True)

# One hot encoding the 'sex' attribute
labelEnc = LabelEncoder()
male = labelEnc.fit_transform(training_set['sex'])
oneHotEnc = OneHotEncoder(categorical_features=[0])
male = oneHotEnc.fit_transform(male.reshape(-1, 1)).toarray()

# Inserting the one hot encoding attribute and dropping the 'sex' attribute
training_set = training_set.drop(labels='sex', axis=1)
training_set.insert(training_set.shape[1], "male", male[:, 0], True) 
training_set.insert(training_set.shape[1], "female", male[:, 1], True)

# Encoding taxi_type
training_temp = {}
for i in range(len(training_set.taxi_type.unique())):
    training_temp["taxi_type_{}".format(sorted(training_set.taxi_type.unique())[i])] = np.zeros(training_set.shape[0], dtype="float32")

for i, taxi in enumerate(training_set['taxi_type']):
    training_temp['taxi_type_{}'.format(taxi)][i] = 1.0

for key in training_temp.keys():
    training_set[key] = training_temp[key]

training_set = training_set.drop(labels='taxi_type', axis=1)

# For trying label encoding only
# training_set['taxi_type'] = labelEnc.fit_transform(training_set['taxi_type'])

# Encoding customer_score_confidence
training_temp = {}
for i in range(len(training_set.customer_score_confidence.unique())):
    training_temp["customer_score_confidence_{}".format(sorted(training_set.customer_score_confidence.unique())[i])] = np.zeros(training_set.shape[0], dtype="float32")

for i, taxi in enumerate(training_set['customer_score_confidence']):
    training_temp['customer_score_confidence_{}'.format(taxi)][i] = 1.0

for key in training_temp.keys():
    training_set[key] = training_temp[key]

training_set = training_set.drop(labels='customer_score_confidence', axis=1)

# For trying label encoding only
# training_set['customer_score_confidence'] = labelEnc.fit_transform(training_set['customer_score_confidence'])

# Encoding drop_location_type
training_temp = {}
for i in range(len(training_set.drop_location_type.unique())):
    training_temp["drop_location_type_{}".format(sorted(training_set.drop_location_type.unique())[i])] = np.zeros(training_set.shape[0], dtype="float32")

for i, taxi in enumerate(training_set['drop_location_type']):
    training_temp['drop_location_type_{}'.format(taxi)][i] = 1.0

for key in training_temp.keys():
    training_set[key] = training_temp[key]

training_set = training_set.drop(labels='drop_location_type', axis=1)

# print(training_set)

training_set1 = training_set

# Replacing nan in annon_var_1 with mean
training_set['anon_var_1'].fillna(training_set['anon_var_1'].mean(), inplace=True)
# print(training_set)

# Trying dropping the anon_var_1 attribute in training_set1
training_set1 = training_set1.drop(labels='anon_var_1', axis=1)

"""
Doing the same preprocessing for the test data
"""

test_set = pd.read_csv("test.csv")

test_id = test_set['id']
test_id = np.asarray(test_id)

# Dropping id column
test_set = test_set.drop(labels='id', axis=1)

test_set['taxi_type'].fillna('O', inplace=True)

test_set['customer_score'].fillna(test_set['customer_score'].mean(), inplace=True)

test_set['customer_score_confidence'].fillna('O', inplace=True)

test_set['months_of_activity'].fillna(0.0, inplace=True)

labelEnc = LabelEncoder()
male = labelEnc.fit_transform(test_set['sex'])
oneHotEnc = OneHotEncoder(categorical_features=[0])
male = oneHotEnc.fit_transform(male.reshape(-1, 1)).toarray()

test_set = test_set.drop(labels='sex', axis=1)
test_set.insert(test_set.shape[1], "male", male[:, 0], True) 
test_set.insert(test_set.shape[1], "female", male[:, 1], True)

test_temp = {}
for i in range(len(test_set.taxi_type.unique())):
    test_temp["taxi_type_{}".format(sorted(test_set.taxi_type.unique())[i])] = np.zeros(test_set.shape[0], dtype="float32")

for i, taxi in enumerate(test_set['taxi_type']):
    test_temp['taxi_type_{}'.format(taxi)][i] = 1.0

for key in test_temp.keys():
    test_set[key] = test_temp[key]

test_set = test_set.drop(labels='taxi_type', axis=1)

# test_set['taxi_type'] = labelEnc.fit_transform(test_set['taxi_type'])

test_temp = {}
for i in range(len(test_set.customer_score_confidence.unique())):
    test_temp["customer_score_confidence_{}".format(sorted(test_set.customer_score_confidence.unique())[i])] = np.zeros(test_set.shape[0], dtype="float32")

for i, taxi in enumerate(test_set['customer_score_confidence']):
    test_temp['customer_score_confidence_{}'.format(taxi)][i] = 1.0

for key in test_temp.keys():
    test_set[key] = test_temp[key]

test_set = test_set.drop(labels='customer_score_confidence', axis=1)

# test_set['customer_score_confidence'] = labelEnc.fit_transform(test_set['customer_score_confidence'])

test_temp = {}
for i in range(len(test_set.drop_location_type.unique())):
    test_temp["drop_location_type_{}".format(sorted(test_set.drop_location_type.unique())[i])] = np.zeros(test_set.shape[0], dtype="float32")

for i, taxi in enumerate(test_set['drop_location_type']):
    test_temp['drop_location_type_{}'.format(taxi)][i] = 1.0

for key in test_temp.keys():
    test_set[key] = test_temp[key]

test_set = test_set.drop(labels='drop_location_type', axis=1)

test_set1 = test_set
# print(test_set)

test_set['anon_var_1'].fillna(test_set['anon_var_1'].mean(), inplace=True)

test_set1 = test_set1.drop(labels='anon_var_1', axis=1)

# For finiding error on part of train data
# X_train, X_test, y_train, y_test = train_test_split(training_set, training_labels, test_size=0.2, random_state=42)

"""
Preprocessing complete
"""

xg_classify = XGBClassifier(objective='multi:softmax', num_class=3, colsample_bytree=0.8, subsample=0.8, scale_pos_weight=1, learning_rate=0.06, max_depth=5, n_estimators=500, gamma=5)

# Trying data normalization

# sc = StandardScaler()
# sc.fit_transform(training_set)
# sc.fit_transform(test_set)

xg_classify.fit(training_set, training_labels)
print("Data fitting completed")

# Mean Squared Error on the training data
print("mse =", mean_squared_error(training_labels, xg_classify.predict(training_set)))

ans = xg_classify.predict(test_set)
print("Data prediction completed")

# print(test_id.shape)
# print(ans.shape)

# a = accuracy_score(y_test, ans)  
# print("mean squeared : " , a)

print(ans)

# Writing output to the csv
with open("output-xgboost.csv", "w") as fp:
    fp.write("id,pricing_category\n")
    for i in range(test_id.shape[0]):
        fp.write("{},{}.0\n".format(test_id[i], ans[i]))
