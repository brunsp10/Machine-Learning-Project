# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:03:24 2023

@author: bruns
"""

import pandas as pd
import numpy as np

def averaged_perceptron(data, r, T):
    w = np.zeros(len(data[0]) - 1)
    a = np.zeros(len(data[0]) - 1)
    for iteration in range(T):
        np.random.shuffle(data)
        x = data[:,:-1]
        y = data[:,-1]
        for i in range(len(data)):
            if y[i] * (w.T@x[i]) <= 0:
                w = w + (r * y[i]*x[i])
            else:
                a = a + w
    return a

def make_predictions_single_vector(data, w):
    predictions = []
    x = data[:,:-1]
    for i in range(len(data)):
        result = (w.T@x[i])
        if result >= 0:
            predictions.append(1)
        else:
            predictions.append(-1)
    return predictions

def calculate_error(data, predictions):
    count = 0
    actual = data[:,-1]
    for i in range(len(predictions)):
        if actual[i] != predictions[i]:
            count += 1
    return count/len(data)

def main():
    # Read in the data
    train_df = pd.read_csv("train_final_gd.csv")
    train_array = np.array(train_df)
    train_array = np.insert(train_array,0,1,axis=1)
    for i in range(len(train_array)):
        if train_array[i][-1] == "yes":
            train_array[i][-1] = 1
        else:
            train_array[i][-1] = -1
            
    # Setup the splits
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
    
    # Make lists of arrays for easier iterating
    train_sets = [train_1, train_2, train_3, train_4, train_5]
    test_sets = [test_1, test_2, test_3, test_4, test_5]
    
    r_s = np.linspace(0.001, 0.5, 500)
    best_r = 0.001
    best_error = 1
    for r in r_s:
        print(f"r = {r}")
        errors = []
        for i in range(len(train_sets)):
            train_array = train_sets[i]
            test_array = test_sets[i]
            w = averaged_perceptron(train_array, r, 100)
            predicted_labels = make_predictions_single_vector(test_array, w)
            errors.append(calculate_error(test_array, predicted_labels))
        avg_error = sum(errors)/5
        print(f"Average error = {avg_error}")
        if avg_error < best_error:
            best_error = avg_error
            best_r = r
    print(f"The best value of r is {best_r}")

if __name__=="__main__":
    main()