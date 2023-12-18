import numpy as np
import pandas as pd
from math import exp as exp
from datetime import datetime

def stochastic_gradient_descent(data,gamma,d,width):
    t=0
    output_weights = np.random.normal(0,1,width)
    hidden_layer_weights = np.random.normal(0,1,size = (width,width))
    input_weights = np.random.normal(0,1,size = (len(data[0])-1,width))
    for iteration in range(1,101):
        np.random.shuffle(data)
        for i in range(len(data)):
            sample = data[i]
            predicted_y, hidden_layer_1_z, hidden_layer_2_z = forward_pass(sample, input_weights, hidden_layer_weights, output_weights, width)
            if predicted_y == data[i][-1]:
                continue
            else:
                gamma_t = gamma/(1+(gamma/d)*t)
                t+=1
                output_weight_gradient, hidden_layer_weights_gradient,input_weights_gradient = backpropogation(predicted_y, sample, input_weights, hidden_layer_weights,
                                  output_weights, hidden_layer_1_z, hidden_layer_2_z, width)
                output_weights = output_weights - gamma_t*output_weight_gradient
                input_weights = input_weights - gamma_t*input_weights_gradient
                hidden_layer_weights = hidden_layer_weights - gamma_t*hidden_layer_weights_gradient  
    return input_weights, hidden_layer_weights, output_weights


def backpropogation(predicted_y, sample, input_weights, hidden_layer_weights, output_weights,
                      hidden_layer_1_z, hidden_layer_2_z, width):
    x = sample[:-1]
    y = sample[-1]
    derivatives = {}
    output_weight_gradient = np.zeros(width)
    hidden_layer_weights_gradient = np.zeros(shape = (width,width))
    input_weights_gradient = np.zeros(shape = (len(x),width))
    
    # Start caching derivatives
    derivatives["dL/dy"] = predicted_y - y
    
    # Fill in for the output layer
    for i in range(len(output_weights)):
        derivative_string = f"dL/dw^3_{i}"
        derivatives[derivative_string] = derivatives["dL/dy"] * hidden_layer_2_z[i]
        output_weight_gradient[i] = derivatives["dL/dy"] * hidden_layer_2_z[i]
    
    # Compute derivatives for z^2 layer
    for i in range(len(hidden_layer_2_z)):
        derivative_string = f"dL/dz^2_{i}"
        derivatives[derivative_string] = derivatives["dL/dy"] * output_weights[i]
    
    # Fill in for hidden layer weights
    for i in range(len(hidden_layer_2_z)):
        
        # First z is a constant with no incoming edges
        for j in range(1,len(hidden_layer_2_z)):
            derivative_string = f"dL/dw^2_{i}{j}"
            grad = derivatives[f"dL/dz^2_{j}"] * hidden_layer_1_z[i] * hidden_layer_2_z[j]*(1-hidden_layer_2_z[j])
            derivatives[derivative_string] = grad
            hidden_layer_weights_gradient[i][j] = grad
          
    # Compute derivatives for z^1 layer
    for i in range(len(hidden_layer_1_z)):
        derivative_string = f"dL/dz^1_{i}"
        grad = 0
        
        # Loop over all possible next visited nodes
        for k in range(1,len(output_weights)):
            next_node_string = f"dL/dz^2_{k}"
            next_node_derivative = derivatives[next_node_string]
            grad = grad + next_node_derivative * hidden_layer_weights[i][k] * hidden_layer_2_z[j] * (1-hidden_layer_2_z[j])
        derivatives[derivative_string] = grad
          
    # Fill in for input layer weights
    for i in range(len(x)):
        
        # First z is a constant with no incoming edges
        for j in range(1,len(hidden_layer_1_z)):
            grad = derivatives[f"dL/dz^1_{j}"]*x[i]*hidden_layer_1_z[j]*(1-hidden_layer_1_z[j])
            input_weights_gradient[i][j] = grad
    return output_weight_gradient, hidden_layer_weights_gradient,input_weights_gradient

def forward_pass(sample, input_weights, hidden_layer_weights, output_weights, width):
    hidden_layer_1_z = np.zeros(width)
    hidden_layer_2_z = np.zeros(width)
    x = sample[:-1]
    hidden_layer_1_z[0] = 1
    hidden_layer_2_z[0] = 1    
    
    # Get the values for first hidden layer
    for i in range(1,width):
        for j in range(len(x)):
            hidden_layer_1_z[i] += x[j] * input_weights[j][i]
    for i in range(1,width):
        hidden_layer_1_z[i] = sigmoid(hidden_layer_1_z[i])
    
    # Get the raw values for second hidden layer
    for i in range(1,width):
        for j in range(width):
            hidden_layer_2_z[i] += hidden_layer_1_z[j] * hidden_layer_weights[j][i]
    for i in range(1,width):
        hidden_layer_2_z[i] = sigmoid(hidden_layer_2_z[i])
    
    # Get the output value
    result = 0
    for i in range(width):
        result += hidden_layer_2_z[i] * output_weights[i]
    return result, hidden_layer_1_z, hidden_layer_2_z

def sigmoid(x):
    try:
        result = 1/(1+exp(-1*x))
    except:
        if x > 0:
            result = 1
        else:
            result = 0
    return result

def make_predictions(data, input_weights, hidden_layer_weights, output_weights,width):
    predictions = []
    for i in range(len(data)):
        predicted_y, hidden_layer_1_z, hidden_layer_2_z = forward_pass(data[i], input_weights, hidden_layer_weights, output_weights, width)
        predictions.append(predicted_y)
    for j in range(len(predictions)):
        if predictions[j] >= 0:
            predictions[j] = 1
        else:
            predictions[j] = -1
    return predictions

def calculate_error(predictions,data):
    count = 0
    actual = data[:,-1]
    for i in range(len(predictions)):
        if actual[i] != predictions[i]:
            count += 1
    return count/len(data)

def main():
    train_df = pd.read_csv("train_with_ada.csv")
    train_array = np.array(train_df)
    train_array = np.insert(train_array,0,1,axis=1)
    for i in range(len(train_array)):
        if train_array[i][-1] == "yes":
            train_array[i][-1] = 1
        else:
            train_array[i][-1] = -1
    train_array = train_array.astype('float')    
    
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
    
    # Run the iterations
    d = 1
    gammas = [0.0001,
              0.0005,
              0.001,
              0.001888888888888889,
              0.005,
              0.01,
              0.05,
              0.1,
              0.5,
              1]
    best_error = 1
    best_gamma = gammas[0]
    for gamma in gammas:
        print(f"Running for gamma = {gamma}")
        errors = 0
        for i in range(len(train_sets)):
            print(i)
            t1 = datetime.now()
            train_array = train_sets[i]
            test_array = test_sets[i]
            input_weights, hidden_layer_weights, output_weights = stochastic_gradient_descent(train_array, gamma, d, 5)
            test_predictions = make_predictions(test_array,input_weights, hidden_layer_weights, output_weights,5)
            test_error = calculate_error(test_predictions, test_array)
            errors += test_error
            t2 = datetime.now()
            print(t2-t1)
        avg_error = errors/5
        if avg_error < best_error:
            best_error = avg_error
            best_gamma = gamma
    
    print(f"The best gamma is {best_gamma}")
    #The best gamma is 0.001888888888888889 is Ada-Boost not included

if __name__ == '__main__':
    main()