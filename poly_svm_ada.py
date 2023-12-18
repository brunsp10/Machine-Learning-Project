import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from datetime import datetime
#%% Model run with my params
print("Running with cv parameters")
train_df = pd.read_csv("train_final_gd.csv")
test_df = pd.read_csv("test_final_gd.csv")
train_array = np.array(train_df)
test_array = np.array(test_df)
train_array = np.insert(train_array,0,1,axis=1)
test_array = np.insert(test_array,0,1,axis=1)
for i in range(len(train_array)):
    if train_array[i][-1] == "yes":
        train_array[i][-1] = 1
    else:
        train_array[i][-1] = -1
train_array = train_array.astype('float')
test_array = test_array.astype('float')
clf = make_pipeline(StandardScaler(), SVC(C=0.8018327605956472, kernel = "poly",
                                          degree = 3, gamma=1))
x = train_array[:,:-1]
y = train_array[:,-1]
clf.fit(x,y)
predictions = clf.predict(test_array)
for j in range(len(predictions)):
    if predictions[j] == -1:
        predictions[j] = 0
ids = list(range(1,23843))
results = {'ID':ids, 'Prediction':predictions}
result_df = pd.DataFrame.from_dict(results)
result_df.to_csv("./results/poly_svm.csv.csv", index = False)
#%% Model run with defaults
print("Running with default parameters")
train_df = pd.read_csv("train_final_gd.csv")
test_df = pd.read_csv("test_final_gd.csv")
train_array = np.array(train_df)
test_array = np.array(test_df)
train_array = np.insert(train_array,0,1,axis=1)
test_array = np.insert(test_array,0,1,axis=1)
for i in range(len(train_array)):
    if train_array[i][-1] == "yes":
        train_array[i][-1] = 1
    else:
        train_array[i][-1] = -1
train_array = train_array.astype('float')
test_array = test_array.astype('float')
clf = make_pipeline(StandardScaler(), SVC(kernel="poly"))
x = train_array[:,:-1]
y = train_array[:,-1]
clf.fit(x,y)
predictions = clf.predict(test_array)
for j in range(len(predictions)):
    if predictions[j] == -1:
        predictions[j] = 0
ids = list(range(1,23843))
results = {'ID':ids, 'Prediction':predictions}
result_df = pd.DataFrame.from_dict(results)
result_df.to_csv("./results/poly_svm_defaults.csv.csv", index = False)
#%% Model run with my params
print("Running with cv parameters with AB score added")
train_df = pd.read_csv("train_with_ada.csv")
test_df = pd.read_csv("test_with_ada.csv")
train_array = np.array(train_df)
test_array = np.array(test_df)
train_array = np.insert(train_array,0,1,axis=1)
test_array = np.insert(test_array,0,1,axis=1)
for i in range(len(train_array)):
    if train_array[i][-1] == "yes":
        train_array[i][-1] = 1
    else:
        train_array[i][-1] = -1
train_array = train_array.astype('float')
test_array = test_array.astype('float')
clf = make_pipeline(StandardScaler(), SVC(C=0.8018327605956472, kernel = "poly",
                                          degree = 3, gamma=1))
x = train_array[:,:-1]
y = train_array[:,-1]
clf.fit(x,y)
predictions = clf.predict(test_array)
for j in range(len(predictions)):
    if predictions[j] == -1:
        predictions[j] = 0
ids = list(range(1,23843))
results = {'ID':ids, 'Prediction':predictions}
result_df = pd.DataFrame.from_dict(results)
result_df.to_csv("./results/poly_svm_ada.csv", index = False)
#%% Model run with defaults
print("Running with default parameters with AB score added")
train_df = pd.read_csv("train_with_ada.csv")
test_df = pd.read_csv("test_with_ada.csv")
train_array = np.array(train_df)
test_array = np.array(test_df)
train_array = np.insert(train_array,0,1,axis=1)
test_array = np.insert(test_array,0,1,axis=1)
for i in range(len(train_array)):
    if train_array[i][-1] == "yes":
        train_array[i][-1] = 1
    else:
        train_array[i][-1] = -1
train_array = train_array.astype('float')
test_array = test_array.astype('float')
clf = make_pipeline(StandardScaler(), SVC(kernel="poly"))
x = train_array[:,:-1]
y = train_array[:,-1]
clf.fit(x,y)
predictions = clf.predict(test_array)
for j in range(len(predictions)):
    if predictions[j] == -1:
        predictions[j] = 0
ids = list(range(1,23843))
results = {'ID':ids, 'Prediction':predictions}
result_df = pd.DataFrame.from_dict(results)
result_df.to_csv("./results/poly_svm_defaults_ada.csv", index = False)