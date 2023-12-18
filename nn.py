import numpy as np
import pandas as pd
from math import exp as exp
from datetime import datetime

def stochastic_gradient_descent(data,gamma,d,width):
    t=0
    output_weights = np.random.normal(0,1,width)
    hidden_layer_weights = np.random.normal(0,1,size = (width,width))
    input_weights = np.random.normal(0,1,size = (len(data[0])-1,width))
    for iteration in range(1,1001):
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

def make_predictions_test(data, input_weights, hidden_layer_weights, output_weights,width):
    predictions = []
    for i in range(len(data)):
        predicted_y, hidden_layer_1_z, hidden_layer_2_z = forward_pass_test(data[i], input_weights, hidden_layer_weights, output_weights, width)
        predictions.append(predicted_y)
    for j in range(len(predictions)):
        if predictions[j] >= 0:
            predictions[j] = 1
        else:
            predictions[j] = -1
    return predictions

def forward_pass_test(sample, input_weights, hidden_layer_weights, output_weights, width):
    hidden_layer_1_z = np.zeros(width)
    hidden_layer_2_z = np.zeros(width)
    x = sample
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


def calculate_error(predictions,data):
    count = 0
    actual = data[:,-1]
    for i in range(len(predictions)):
        if actual[i] != predictions[i]:
            count += 1
    return count/len(data)

def main():
    train_df = pd.read_csv("train_with_ada.csv")
    test_df = pd.read_csv("test_with_ada.csv")
    train_array = np.array(train_df)
    test_array = np.array(test_df)
    for i in range(len(train_array)):
        if train_array[i][-1] == "yes":
            train_array[i][-1] = 1
        else:
            train_array[i][-1] = -1
    train_array = train_array.astype('float')
    test_array = test_array.astype('float')
    widths = [5,10,25,50,75,100]
    # Run the iterations
    for width in widths:
        print(f"Running d = 1 for width = {width}")
        t1 = datetime.now()
        d = 1
        gamma = 0.05
        input_weights, hidden_layer_weights, output_weights = stochastic_gradient_descent(train_array, gamma, d, width)
        test_predictions = make_predictions_test(test_array,input_weights, hidden_layer_weights, output_weights,width)
        ids = list(range(1,23843))
        for j in range(len(test_predictions)):
            if test_predictions[j] == -1:
                test_predictions[j] = 0
        results = {'ID':ids, 'Prediction':test_predictions}
        result_df = pd.DataFrame.from_dict(results)
        result_df.to_csv(f"./results/nn_{width}_1.csv", index = False)
        print(datetime.now() - t1)
    
if __name__ == '__main__':
    main()