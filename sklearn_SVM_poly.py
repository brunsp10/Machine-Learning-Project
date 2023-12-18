import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from datetime import datetime

#%% Cross validation
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
gammas = np.linspace(0.01,1, num = 15)
c_s = [700/873,1,973/873]
degrees = [2,3]

train_1 = train_array[5000:]
test_1 = train_array[:5000]

train_2 = np.concatenate([train_array[:5000], train_array[10000:]])
test_2 = train_array[5000:10000]

train_3 = np.concatenate([train_array[:10000], train_array[15000:]])
test_3 = train_array[10000:15000]

train_4 = np.concatenate([train_array[:15000], train_array[20000:]])
test_4 = train_array[15000:20000]

train_5 = train_array[:20000]
test_5 = train_array[20000:]

train_sets = [train_1, train_2, train_3, train_4, train_5]
test_sets = [test_1, test_2, test_3, test_4, test_5]

best_error = 1
best_degree = degrees[0]
best_c = c_s[0]
best_gamma = gammas[0]

for g in gammas:
    for c in c_s:
        for d in degrees:
            errors = 0
            print(f"Running for gamma = {g}, degree = {d} and c = {c}")
            for i in range(1):
                t1 = datetime.now()
                print(i)
                train = train_sets[i]
                x = train[:,:-1]
                y = train[:,-1]
                test = test_sets[i]
                test_x = test[:,:-1]
                test_y = test[:,-1]
                clf = make_pipeline(StandardScaler(), SVC(C=c, kernel = "poly",
                                                          degree = d, gamma=g))
                clf.fit(x,y)
                predictions = clf.predict(test_x)
                count = 0
                for j in range(len(predictions)):
                    if predictions[j] != test_y[j]:
                        count += 1
                error = count/len(predictions)
                errors += error
                t2 = datetime.now()
                print(t2-t1)
            avg_error = errors
            if avg_error < best_error:
                best_error = avg_error
                best_c = c
                best_degree = d
                best_gamma = g

print(f"The best gamma is {best_gamma}")
print(f"The best c is {best_c}")
print(f"The best degree is {best_degree}")
'''
The best gamma is 1.0
The best c is 0.8018327605956472
The best degree is 3
'''
#%% Model run with my params
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
ids = list(range(1,23843))
results = {'ID':ids, 'Prediction':predictions}
result_df = pd.DataFrame.from_dict(results)
result_df.to_csv("./results/poly_svm.csv.csv", index = False)
#%% Model run with defaults
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
ids = list(range(1,23843))
results = {'ID':ids, 'Prediction':predictions}
result_df = pd.DataFrame.from_dict(results)
result_df.to_csv("./results/poly_svm_defaults.csv.csv", index = False)