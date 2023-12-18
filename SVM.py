from scipy.optimize import minimize as minimize
from scipy.optimize import Bounds as Bounds
from math import exp as exp
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
            matrix[i,j] = gaussian(x[i],x[j],gamma)*y[i]*y[j]
    return matrix

def gaussian(x1, x2, gamma):
    return exp(-1*((np.linalg.norm(x1-x2))**2)/gamma)

def make_predictions_cv(new_data, train_x, train_y, alphas, b,gamma):
    predictions = []
    new_x = new_data[:,:-1]
    for i in range(len(new_x)):
        total = 0
        current_sample = new_x[i]
        for j in range(len(train_x)):
            total += alphas[j]*train_y[j]*gaussian(current_sample,train_x[j],gamma)
        if total + b >= 0:
            predictions.append(1)
        else:
            predictions.append(-1)
    return predictions

def make_predictions(new_data, train_x, train_y, alphas, b,gamma):
    predictions = []
    new_x = new_data
    for i in range(len(new_x)):
        total = 0
        current_sample = new_x[i]
        for j in range(len(train_x)):
            total += alphas[j]*train_y[j]*gaussian(current_sample,train_x[j],gamma)
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

def calculate_b(data, point,alphas,y,gamma):
    total = 0
    for i in range(len(data)):
        total += alphas[i]*y[i]*gaussian(data[i],point,gamma)
    return total

def main():
    # Cross validate for Gaussian
    # Read in the data
    train_df = pd.read_csv("train_final_gd.csv")
    train_array_orig = np.array(train_df)
    train_array_orig = np.insert(train_array_orig,0,1,axis=1)
    for i in range(len(train_array_orig)):
        if train_array_orig[i][-1] == "yes":
            train_array_orig[i][-1] = 1
        else:
            train_array_orig[i][-1] = -1
    
    # Make list of values for gamma
    gammas = np.linspace(1,50, num = 10)
    c_s = [100/873,500/873,700/873]
    best_error = 1
    best_c = c_s[0]
    best_gamma = gammas[0]
    train_array = train_array_orig[:20000]
    test_array = train_array_orig[20000:]
    x = train_array[:,:-1]
    y = train_array[:,-1]
    for gamma in gammas:
        for c in c_s:
            print(f"Running for gamma = {gamma} and C = {c:0.3f}")
            gram_matrix = build_matrix(x, y, gamma)
            print("Built matrix")
            a = np.zeros(len(gram_matrix))
            cons = ({'type':'eq','fun': lambda a: np.dot(a,y)})
            bnds = Bounds(0,c)
            result = minimize(dual_svm, a,args = (gram_matrix),
                              method = "SLSQP",bounds = bnds,constraints=cons)
            print("Finished opitimization")
            alphas = result.x
            b_s = []
            for j in range(len(x)):
                if alphas[j] > 0 and alphas[j] < c:
                    b_s.append(y[j] - calculate_b(x,x[j],alphas,y,gamma))
            b = sum(b_s)/len(b_s)
            test_predictions = make_predictions_cv(test_array,x,y,alphas,b,gamma)
            test_error = calculate_error(test_array, test_predictions)
            if test_error < best_error:
                best_c = c
                best_gamma = gamma
                best_error = test_error
    
    print(f"The best gamma is {best_gamma} and the best c is {best_c}")
    # Run Gaussian SVM
    train_df = pd.read_csv('train_final_gd.csv')
    test_df = pd.read_csv('test_final_gd.csv')
    test_array = test_df.to_numpy()
    train_array = train_df.to_numpy()
    for i in range(len(train_array)):
        if train_array[i][-1] == "yes":
            train_array[i][-1] = 1
        else:
            train_array[i][-1] = -1
    train_array = np.insert(train_array,0,1,axis=1)
    test_array = np.insert(test_array,0,1,axis=1)
    x = train_array[:,:-1]
    y = train_array[:,-1]
    gram_matrix = build_matrix(x,y,best_gamma)
    a = np.zeros(len(x))
    cons = ({'type':'eq','fun': lambda a: np.dot(a,y)})
    bnds = Bounds(0,best_c)
    result = minimize(dual_svm, a,args = (gram_matrix),
                      method = "SLSQP",bounds = bnds,constraints=cons)
    alphas = result.x
    b_s = []
    for j in range(len(x)):
        if alphas[j] > 0 and alphas[j] < best_c:
            b_s.append(y[j] - calculate_b(x,x[j],alphas,y,best_gamma))
    b = sum(b_s)/len(b_s)
    test_predictions = make_predictions(test_array,x,y,alphas,b,best_gamma)
    
    # Save results
    for i in range(len(test_predictions)):
        if test_predictions[i] == -1:
            test_predictions[i] = 0
    ids = list(range(1,23843))
    results = {'ID':ids, 'Prediction':test_predictions}
    result_df = pd.DataFrame.from_dict(results)
    result_df.to_csv("./results/gaussian_SVM.csv", index = False)
    
    
if __name__=="__main__":
    main()