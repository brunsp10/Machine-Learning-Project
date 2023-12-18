import numpy as np
import pandas as pd
from math import exp as exp
from matplotlib import pyplot as plt

def stochastic_gradient_descent(data,gamma,d,width):
    losses = []
    t=0
    output_weights = np.random.normal(0,1,width)
    hidden_layer_weights = np.random.normal(0,1,size = (width,width))
    input_weights = np.random.normal(0,1,size = (len(data[0])-1,width))
    for iteration in range(1,1001):
        print(iteration)
        np.random.shuffle(data)
        for i in range(len(data)):
            sample = data[i]
            t+=1
            predicted_y, hidden_layer_1_z, hidden_layer_2_z = forward_pass(sample, input_weights, hidden_layer_weights, output_weights, width)
            if predicted_y == data[i][-1]:
                continue
            else:
                gamma_t = gamma/(1+(gamma/d)*t)
                output_weight_gradient, hidden_layer_weights_gradient,input_weights_gradient = backpropogation(predicted_y, sample, input_weights, hidden_layer_weights,
                                  output_weights, hidden_layer_1_z, hidden_layer_2_z, width)
                output_weights = output_weights - gamma_t*output_weight_gradient
                input_weights = input_weights - gamma_t*input_weights_gradient
                hidden_layer_weights = hidden_layer_weights - gamma_t*hidden_layer_weights_gradient  
        total_loss = []
        for i in range(len(data)):
            predicted_y, hidden_layer_1_z, hidden_layer_2_z = forward_pass(data[i], input_weights, hidden_layer_weights, output_weights, width)
            total_loss.append((predicted_y-data[i][-1])**2)
        losses.append(sum(total_loss)*.5)
    return losses


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
        if predictions[j] > 0.5:
            predictions[j] = 1
        else:
            predictions[j] = 0
    return predictions

def calculate_error(predictions,data):
    count = 0
    actual = data[:,-1]
    for i in range(len(predictions)):
        if actual[i] != predictions[i]:
            count += 1
    return count/len(data)

def main():
    # Read in data and convert 0 labels to -1
    train_df = pd.read_csv('train_with_ada.csv')
    train_array = train_df.to_numpy()
    for i in range(len(train_array)):
        if train_array[i][-1] == "yes":
            train_array[i][-1] = 1
        else:
            train_array[i][-1] = -1

    # Add the ones column for constant term in x
    train_array = np.insert(train_array,0,1,axis=1)
    #gamma = 0.001888888888888889
    gamma = 0.05
    d = 1
    
    # Run the iterations
    losses = stochastic_gradient_descent(train_array, gamma, d, 5)
    plt.figure(figsize=(12,8))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot([i for i in range(1,1001)],losses)
    '''
    This was for a test with 150 iterations
    plt.savefig("fig1.png")
    '''
    '''
    This was for a test with 200 iterations
    plt.savefig("fig2.png")
    '''
    '''
    This was for a test with 300 iterations
    plt.savefig("fig3.png")
    '''
    '''
    This was for a test with 500 iterations
    plt.savefig("fig4.png")
    '''
    plt.savefig("fig5.png")

if __name__ == '__main__':
    main()