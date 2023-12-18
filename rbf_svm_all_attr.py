import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#%% Setup the replacement arrays
def set_replace_values(current, a, replacements):
    new_list = []
    for v in current:
        new_list.append(replacements[a][v])
    return new_list

train_df = pd.read_csv("train_final.csv")
test_df = pd.read_csv("test_final.csv")
attribute_list = ['age', 'workclass', 'fnlwgt', 'education','education.num','marital.status',
                  'relationship', 'race', 'sex', 'capital.gain','capital.loss',
                  'hours.per.week','native.country','occupation']
replace_values = {}

for a in attribute_list:
    if train_df[a].dtype != 'int64':
        train_values = set(train_df[a].values)
        test_values = set(test_df[a].values)
        train_values.update(test_values)
        count = 1
        replace_values[a] = {}
        for v in train_values:
            if v == "?":
                replace_values[a][v] = 0
            else:
                replace_values[a][v] = count
                count += 1
        train_df[a] = set_replace_values(train_df[a], a, replace_values)
        test_df[a] = set_replace_values(test_df[a], a, replace_values)

train_array = np.array(train_df)
test_array = np.array(test_df)

for i in range(len(train_array)):
    if train_array[i][-1] == "yes":
        train_array[i][-1] = 1
    else:
        train_array[i][-1] = -1
train_array = train_array.astype('float')
test_array = test_array.astype('float')

#%% Run rbf with default params
clf = make_pipeline(StandardScaler(), SVC())
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
result_df.to_csv("./results/rbf_all_attr_defaults.csv", index = False)
