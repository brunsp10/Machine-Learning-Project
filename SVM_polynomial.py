from scipy.optimize import minimize as minimize
from scipy.optimize import Bounds as Bounds
from math import exp as exp
from math import sqrt as sqrt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def dual_svm(alpha,gram_matrix):
    return .5 * np.dot(gram_matrix @ alpha,alpha) - sum(alpha)

def build_matrix(x,y,gamma):
    matrix = np.zeros(shape = (len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            matrix[i,j] = kernel(x[i],x[j],gamma)*y[i]*y[j]
    return matrix

def kernel(x1, x2,c):
    squared = 0
    for i in range(len(x1)):
        squared += (x1[i]**2) * (x2[i]**2)
    middle = 0
    for i in range(2,len(x1)):
        for j in range(i):
            middle += (sqrt(2)*x1[i]*x1[j])*(sqrt(2)*x2[i]*x2[j])
    last = 0
    for i in range(len(x1)):
        last += (sqrt(2*c)*x1[i])*(sqrt(2*c)*x2[i])
    return squared + middle + last + c**2

def make_predictions(new_data, train_x, train_y, alphas, b,c):
    predictions = []
    new_x = new_data[:,:-1]
    for i in range(len(new_x)):
        total = 0
        current_sample = new_x[i]
        for j in range(len(train_x)):
            total += alphas[j]*train_y[j]*kernel(current_sample,train_x[j],c)
        if total + b >= 0:
            predictions.append(1)
        else:
            predictions.append(-1)
    return predictions

def calculate_error(data, predictions):
    truth = data[:,-1]
    total = 0
    for i in range(len(predictions)):
        if truth[i] != predictions[i]:
            total += 1
    return total/len(predictions)

def calculate_b(data, point,alphas,y,c):
    total = 0
    for i in range(len(data)):
        total += alphas[i]*y[i]*kernel(data[i],point,c)
    return total

def main():
    # Cross validate for Gaussian
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
    
    
    
    # Run Gaussian SVM
    
    
    # Save results
    return 0
    
    
if __name__=="__main__":
    main()