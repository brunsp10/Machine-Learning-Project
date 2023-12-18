import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

train_df = pd.read_csv("train_final_gd.csv")
test_df = pd.read_csv("test_final_gd.csv")

train_array = np.array(train_df)
for i in range(len(train_array)):
    if train_array[i][-1] == "yes":
        train_array[i][-1] = 1
    else:
        train_array[i][-1] = -1
        
kf = KFold(n_splits = 5)
kf.get_n_splits(train_array)

for i, (train_index, test_index) in enumerate(kf.split(train_array)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    
'''
Train 1: 5000 to 24999
Test 1: 0 to 4999

Train 2: 0 to 4999, 10000 to 24999
Test 2: 5000 to 9999

Train 3: 0 to 9999, 15000 to 24999
Test 3: 10000 to 14999

Train 4: 0 to 14999, 20000 to 24999
Test 4: 15000 to 19999

Train 5: 0 to 19999
Test 5: 20000 to 24999
'''

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








