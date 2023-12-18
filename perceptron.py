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
    x = data
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
        
    T_s = [100,250,500,1000, 10_000]
    # Run the perceptron
    for T in T_s:
        print(T)
        w = averaged_perceptron(train_array, 0.135, T)
        predicted_labels = make_predictions_single_vector(test_array, w)
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == -1:
                predicted_labels[i] = 0
        ids = list(range(1,23843))
        results = {'ID':ids, 'Prediction':predicted_labels}
        result_df = pd.DataFrame.from_dict(results)
        result_df.to_csv(f"./results/averaged_perceptron_{T}.csv", index = False)
    
if __name__=="__main__":
    main()