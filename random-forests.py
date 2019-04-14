import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

training_set = pd.read_csv("train.csv")

# Extracting labels from training set
training_labels = training_set['pricing_category']
print(training_labels)

# Dropping the last column and id from training set
training_set = training_set.drop(labels='pricing_category', axis=1)
training_set = training_set.drop(labels='id', axis=1)
# print(training_set)

training_set['taxi_type'].fillna('O', inplace=True)

training_set['customer_score'].fillna(training_set['customer_score'].mean(), inplace=True)

training_set['customer_score_confidence'].fillna('O', inplace=True)

training_set['months_of_activity'].fillna(0.0, inplace=True)

labelEnc = LabelEncoder()
male = labelEnc.fit_transform(training_set['sex'])
oneHotEnc = OneHotEncoder(categorical_features=[0])
male = oneHotEnc.fit_transform(male.reshape(-1, 1)).toarray()
# print(male)

training_temp = {}
for i in range(len(training_set.taxi_type.unique())):
    training_temp["taxi_type_{}".format(sorted(training_set.taxi_type.unique())[i])] = np.zeros(training_set.shape[0], dtype="float32")

for i, taxi in enumerate(training_set['taxi_type']):
    training_temp['taxi_type_{}'.format(taxi)][i] = 1.0

for key in training_temp.keys():
    training_set[key] = training_temp[key]

training_set = training_set.drop(labels='taxi_type', axis=1)

training_temp = {}
for i in range(len(training_set.customer_score_confidence.unique())):
    training_temp["customer_score_confidence_{}".format(sorted(training_set.customer_score_confidence.unique())[i])] = np.zeros(training_set.shape[0], dtype="float32")

for i, taxi in enumerate(training_set['customer_score_confidence']):
    training_temp['customer_score_confidence_{}'.format(taxi)][i] = 1.0

for key in training_temp.keys():
    training_set[key] = training_temp[key]

training_set = training_set.drop(labels='customer_score_confidence', axis=1)

training_temp = {}
for i in range(len(training_set.drop_location_type.unique())):
    training_temp["drop_location_type_{}".format(sorted(training_set.drop_location_type.unique())[i])] = np.zeros(training_set.shape[0], dtype="float32")

for i, taxi in enumerate(training_set['drop_location_type']):
    training_temp['drop_location_type_{}'.format(taxi)][i] = 1.0

for key in training_temp.keys():
    training_set[key] = training_temp[key]

training_set = training_set.drop(labels='drop_location_type', axis=1)

training_set = training_set.drop(labels='sex', axis=1)
training_set.insert(training_set.shape[1], "male", male[:, 0], True) 
training_set.insert(training_set.shape[1], "female", male[:, 1], True)
print(training_set)

training_set1 = training_set

training_set['anon_var_1'].fillna(training_set['anon_var_1'].mean(), inplace=True)
# print(training_set)

training_set1 = training_set1.drop(labels='anon_var_1', axis=1)
# print(training_set1)

test_set = pd.read_csv("test.csv")

# Dropping is column
test_id = test_set['id']
test_id = np.asarray(test_id)

test_set = test_set.drop(labels='id', axis=1)

test_set['taxi_type'].fillna('O', inplace=True)

test_set['customer_score'].fillna(training_set['customer_score'].mean(), inplace=True)

test_set['customer_score_confidence'].fillna('O', inplace=True)

test_set['months_of_activity'].fillna(0.0, inplace=True)

labelEnc = LabelEncoder()
male = labelEnc.fit_transform(test_set['sex'])
oneHotEnc = OneHotEncoder(categorical_features=[0])
male = oneHotEnc.fit_transform(male.reshape(-1, 1)).toarray()
# print(male)

test_temp = {}
for i in range(len(test_set.taxi_type.unique())):
    test_temp["taxi_type_{}".format(sorted(test_set.taxi_type.unique())[i])] = np.zeros(test_set.shape[0], dtype="float32")

for i, taxi in enumerate(test_set['taxi_type']):
    test_temp['taxi_type_{}'.format(taxi)][i] = 1.0

for key in test_temp.keys():
    test_set[key] = test_temp[key]

test_set = test_set.drop(labels='taxi_type', axis=1)

test_temp = {}
for i in range(len(test_set.customer_score_confidence.unique())):
    test_temp["customer_score_confidence_{}".format(sorted(test_set.customer_score_confidence.unique())[i])] = np.zeros(test_set.shape[0], dtype="float32")

for i, taxi in enumerate(test_set['customer_score_confidence']):
    test_temp['customer_score_confidence_{}'.format(taxi)][i] = 1.0

for key in test_temp.keys():
    test_set[key] = test_temp[key]

test_set = test_set.drop(labels='customer_score_confidence', axis=1)

test_temp = {}
for i in range(len(test_set.drop_location_type.unique())):
    test_temp["drop_location_type_{}".format(sorted(test_set.drop_location_type.unique())[i])] = np.zeros(test_set.shape[0], dtype="float32")

for i, taxi in enumerate(test_set['drop_location_type']):
    test_temp['drop_location_type_{}'.format(taxi)][i] = 1.0

for key in test_temp.keys():
    test_set[key] = test_temp[key]

test_set = test_set.drop(labels='drop_location_type', axis=1)

test_set = test_set.drop(labels='sex', axis=1)
test_set.insert(test_set.shape[1], "male", male[:, 0], True) 
test_set.insert(test_set.shape[1], "female", male[:, 1], True) 

test_set1 = test_set
print(test_set)

test_set['anon_var_1'].fillna(test_set['anon_var_1'].mean(), inplace=True)

test_set1 = test_set1.drop(labels='anon_var_1', axis=1)

print(training_labels.sum(axis=0))

# Using sklearn random forest classifier
clf = RandomForestClassifier(n_estimators=1000)

# Fitting the training data
clf.fit(training_set, training_labels)
print("Data fitting completed")

ans = clf.predict(test_set)
print("Data prediction completed")

print(test_id.shape)
print(ans.shape)

print(ans)

with open("output-random-forests.csv", "w") as fp:
    fp.write("id,pricing_category\n")
    for i in range(test_id.shape[0]):
        fp.write("{},{}.0\n".format(test_id[i], ans[i]))
