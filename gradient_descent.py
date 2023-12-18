import pandas as pd
import numpy as np
from random import randint
import argparse
import matplotlib.pyplot as plt


def batch_gradient_descent(data, w, r):
    x = data[:,:-1]
    y = data[:,-1]
    w_t = np.zeros(5)
    tain_errors = []
    iterations = [0]
    iteration = 1
    while True:
        print(iteration)
        print(w)
        sum_difference = np.zeros(5)
        w = w_t
        for i in range(len(x)):
            predicted_y = np.matrix.transpose(w)@x[i]
            diff = (y[i] - predicted_y)
            sum_difference -= diff * x[i]
        w_t = w - r * sum_difference
        iterations.append(iteration)
        iteration += 1
        if np.linalg.norm(w - w_t,2) < 1 * 10**-6:
            break
    return w_t, iterations, tain_errors

def stochastic_gradient_descent(data, w, r):
    w_t = np.zeros(5)
    x = data[:,:-1]
    y = data[:,-1]
    tain_errors = []
    iterations = [0]
    iteration = 1
    while iteration < 10_000_000:
        print(iteration)
        print(w)
        w = w_t
        i = randint(0, len(data)-1)
        sum_difference = np.zeros(5)
        predicted_y = np.matrix.transpose(w)@x[i]
        diff = (y[i] - predicted_y)
        sum_difference -= diff * x[i]
        w_t = w - r * sum_difference
        iterations.append(iteration)
        iteration += 1
    return w_t, iterations, tain_errors

def direct_calculation(data):
    x = data[:,:-1]
    y = data[:,-1]
    return np.linalg.inv(np.matrix.transpose(x)@x) @ np.matrix.transpose(x) @ y

def calculate_squared_error(data, w):
    predictions = []
    for i in range(len(data)):
        x = data[i]
        predicted_y = np.matrix.transpose(w)@x
        if predicted_y <= 0.5:
            predictions.append(0)
        else:
            predictions.append(1)
    return predictions

def main():
    # Read in the data
    train_df = pd.read_csv('train_0_1_label_gd_reduced.csv')
    test_df = pd.read_csv('test_final_gd_reduced.csv')
    train_array = np.array(train_df)
    test_array = np.array(test_df)
    
    # Add the ones column
    train_array = np.insert(train_array,0,1,axis=1)
    test_array = np.insert(test_array,0,1,axis=1)
    
    # Get the version of gradient calculation
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--calculator", default = "batch")
    parser.add_argument("-r", "--r", default = 0.1)
    args = parser.parse_args()
    calculation_version = args.calculator
    r = float(args.r)
    
    # Initialize w
    w = np.zeros(5)
    
    # Choose the method for the calculation
    if calculation_version == 'batch':
        w, x, y = batch_gradient_descent(train_array, w, r)
        # Calculate the error
        test_predictions = calculate_squared_error(test_array, w)
        ids = list(range(1,23843))
        results = {'ID':ids, 'Prediction':test_predictions}
        result_df = pd.DataFrame.from_dict(results)
        result_df.to_csv("./results/batch.csv", index = False)
    elif calculation_version == 'stochastic':
        w, x, y = stochastic_gradient_descent(train_array, w, r)
        # Calculate the error
        test_predictions = calculate_squared_error(test_array, w)
        ids = list(range(1,23843))
        results = {'ID':ids, 'Prediction':test_predictions}
        result_df = pd.DataFrame.from_dict(results)
        result_df.to_csv("./results/stochastic.csv", index = False)
    elif calculation_version == 'direct':
        w = direct_calculation(train_array)
        # Calculate the error
        test_predictions = calculate_squared_error(test_array, w)
        ids = list(range(1,23843))
        results = {'ID':ids, 'Prediction':test_predictions}
        result_df = pd.DataFrame.from_dict(results)
        result_df.to_csv("./results/direct_reduced.csv", index = False)
    else:
        raise Exception("Invalid calculation version. Supports 'batch' or 'stochastic'")
    
        
if __name__ == '__main__':
    main()
