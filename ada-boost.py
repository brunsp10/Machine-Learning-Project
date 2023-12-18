import pandas as pd
from math import log2, log, exp
import numpy as np

"""
Class used to represent an ID3 decision tree
"""
class ID3_Tree():
    
    def __init__(self):
        self.root = None
        self.depth = 0
    
    """
    Method used to build a decision tree using ID3 and return the root
    """
    def build_tree(self, attributes, data, purity_measure, depth, max_depth, full_data):
        # If all attributes have been used or max depth reached then
        # return the most common label
        if len(attributes) == 0 or depth == max_depth:
            yes_df = data[data['label']=='yes']
            no_df = data[data['label']=='no']
            if sum(yes_df['weight']) >= sum(no_df['weight']):
                return Node('yes')
            return Node('no')
        
        # If all labels are the same then return that label
        elif len(data['label'].unique()) == 1:
            return Node(data['label'].unique()[0])
        
        # Otherwise, recursively build the tree
        else:
            
            # Pick the best attribute for splitting the data
            best_attribute = None
            best_gain = -1
            calculate_current = Purity_Measures(data)
            current_purity = calculate_current.calculate_purity(purity_measure)
            
            # Loop over the attributes and calculate best purity gain
            for a in attributes:
                single_attribute_data = data[[a,'label','weight']]
                calculate_gain = Purity_Measures(single_attribute_data)
                new_purity = calculate_gain.calculate_purity(purity_measure, attribute = a)
                if current_purity - new_purity > best_gain:
                    best_attribute = a
                    best_gain = current_purity - new_purity
            
            # Make a new node with best attribute, set as root if needed
            new_node = Node(best_attribute)
            if self.root is None:
                self.root = new_node
            
            # Recursively call method for all values of best attribute
            values = full_data[best_attribute].unique()
            reduced_attributes = attributes.copy()
            reduced_attributes.remove(best_attribute)
            for v in values:
                reduced_data = data[data[best_attribute] == v]
                if len(reduced_data) == 0:
                    new_node.branches[v] = Node(data.mode()['label'][0])
                else:
                    new_node.branches[v] = self.build_tree(reduced_attributes, 
                                                           reduced_data, 
                                                           purity_measure,
                                                           depth + 1,
                                                           max_depth,
                                                           full_data)
            
            # Return the node created above
            self.depth = depth + 1
            return new_node
    
    """
    Method used to traverse a decision tree and return a prediction
    """
    def traverse_tree(self, data):
        current_node = self.root
        while len(current_node.branches) != 0:
            
            # Get the next node from the branch corresponding to
            # tested attribute value
            attribute = current_node.attribute
            value = data[attribute]
            current_node = current_node.branches[value]
        
        # When a leaf node is reached, return the label
        prediction = current_node.attribute            
        return prediction
    
    """
    Driver method used to make predictions for a dataframe of items
    """
    def make_predictions(self, data):
        # Make list to track predictions
        predicted_labels = []
        
        # Loop over items and use traverse tree to add
        # predictions to the list
        for i in range(0,len(data)):
            current_item = data.loc[i]
            prediction = self.traverse_tree(current_item)
            predicted_labels.append(prediction)
        return predicted_labels
    
    
    """
    Update the weight for tree building
    """
    def update_weights(self, data, predictions, alpha):
        for i in range(len(data)):
            current_item = data.loc[i]
            if current_item['label'] != predictions[i]:
                data.at[i,'weight'] = current_item['weight'] * exp(alpha)
            else:
                data.at[i,'weight'] = current_item['weight'] * exp(-1*alpha)
        weight_sum = sum(data['weight'])
        if weight_sum != 1:
            data['weight'] = data['weight'].apply(lambda x : x / weight_sum)
    
    """
    Calculate the stump error
    """
    def calculate_stump_alpha(self, train_predictions, train_df):
        total_error = 0
        for i in range(len(train_df)):
            current_item = train_df.loc[i]
            if current_item['label'] != train_predictions[i]:
                total_error += current_item['weight']
        return total_error
    
    """
    Calculates the proportion of misclassified items and returns error
    """
    def calculate_error(self, data, predictions):
        correct = 0
        incorrect = 0
        truth = list(data['label'])
        for i in range(0, len(truth)):
            if truth[i] == predictions[i]:
                correct += 1
            else:
                incorrect += 1
        return incorrect/(correct + incorrect)
    
    """
    Make the total prediction
    """
    def make_total_predictions(self, prediction_lists, alphas):
        predicted_results = []
        for i in range(23842):
            sum_predictions = 0
            for j in range(len(prediction_lists)):
                if prediction_lists[j][i] == 'yes':
                    sum_predictions += alphas[j]
                elif prediction_lists[j][i] == 'no':
                    sum_predictions += -1 * alphas[j]
                else:
                    raise Exception("invalid label")
            if sum_predictions >= 0:
                predicted_results.append('yes')
            else:
                predicted_results.append('no')
        return predicted_results


"""
Class used to hold methods for calculating purity by a variety of methods
"""
class Purity_Measures():

    def __init__(self, data):
        self.data = data
    
    """
    Calculates entropy for a dataframe
    """
    def calculate_entropy(self, other_data):
        total_entropy = 0
        
        # Get total label weight
        total_weight = sum(other_data['weight'])
        
        # Get the labels and loop over adding entropy for
        # each label
        for label in ['yes', 'no']:
            label_df = other_data[other_data['label'] == label]
            proportion = sum(label_df['weight'])/total_weight
            if proportion < 0.0001:
                total_entropy += 0
            else:
                label_entropy = (-1) * (proportion)*log2(proportion)
                total_entropy += label_entropy
        return total_entropy
    
    """
    Driver method for calculating information gain with entropy above
    """
    def information_gain(self, attribute = None):
        if attribute is None:
            return self.calculate_entropy(self.data)
        else:
            
            # If an attribute is set, calculate weighted entropy for each 
            # possible value and return the sum
            attribute_entropy = 0
            for v in self.data[attribute].unique():
                reduced_df = self.data[self.data[attribute] == v]
                prop = sum(reduced_df['weight']) / sum(self.data['weight'])
                attribute_entropy += prop * self.calculate_entropy(reduced_df)
            return attribute_entropy
    
    """
    Method used to calculate the Gini index for a dataframe
    """
    def calculate_gini_index(self, other_data):
        total_gini = 0
        label_counts = other_data.value_counts()
        total_items = len(other_data)
        
        # Calculate the squared proportion for each value and sum
        for label in set(other_data):
            label_items = label_counts[label]
            proportion = label_items/total_items
            label_gini = proportion**2
            total_gini += label_gini
        return 1 - total_gini
    
    """
    Driver method for calculating Gini index
    """
    def gini_index(self, attribute = None):
        if attribute is None:
            return self.calculate_gini_index(self.data)
        else:
            
            # If there is a specified attribute, calculate weighted
            # GI for each value and return sum
            attribute_gini = 0
            for v in self.data[attribute].unique():
                reduced_df = self.data[self.data[attribute] == v]
                prop = len(reduced_df)/len(self.data)
                attribute_gini += prop * self.calculate_gini_index(reduced_df['label'])
            return attribute_gini

    def majority_error(self, attribute = None):
        if attribute is None:
            total_items = len(self.data)
            modal_item_total = max(self.data.value_counts())
            
            # The ME will be 1 minus the proportion of the majority item
            return 1 - (modal_item_total/total_items)
        else:
            
            # If there is a specified attribute, calculate weighted
            # ME for each value and return sum
            attribute_maj_error = 0
            for v in self.data[attribute].unique():
                reduced_df = self.data[self.data[attribute] == v]
                prop = len(reduced_df)/len(self.data)
                total_items = len(reduced_df)
                modal_item_total = max(reduced_df['label'].value_counts())
                attribute_maj_error += prop * (1-modal_item_total/total_items)
            return attribute_maj_error
    
    """
    Driver method for calculating the purity of a given dataframe by
    a given purity measure
    """
    def calculate_purity(self, method, attribute = None):
        if method == 'information_gain':
            return self.information_gain(attribute = attribute)
        elif method == 'gini_index':
            return self.gini_index(attribute = attribute)
        elif method == 'majority_error':
            return self.majority_error(attribute = attribute)
        
"""
Class to represent a node on the decision tree
"""
class Node():
    
    def __init__(self, attribute):
        self.attribute = attribute
        self.branches = {}

def assign_quantiles(data, attr, quantiles):
    results = []
    for i in range(len(data)):
        if data[i] >= quantiles[2]:
            results.append("Fourth")
        elif data[i] >= quantiles[1]:
            results.append("Third")
        elif data[i] >= quantiles[0]:
            results.append("Second")
        else:
            results.append("First")
    return results    
    
def main():
    # Read in the data and set the headers to the proper values
    train_df = pd.read_csv("train_final_ada.csv")
    test_df = pd.read_csv("test_final.csv")

    # Make attribute list and set for predictions
    attribute_list = ['age', 'workclass', 'fnlwgt', 'education','education.num','marital.status',
                      'relationship', 'race', 'sex', 'capital.gain','capital.loss',
                      'hours.per.week','native.country']
    attribute_set = set(attribute_list)
    
    # Replace numerical values with above/below the media
    quantiles = {}
    for a in attribute_list:
        if train_df[a].dtype == 'int64':
            quantiles[a] = np.quantile(train_df[a], [0.25,0.5,0.75])
    for a in attribute_list:
        if train_df[a].dtype == 'int64':
            train_df[a] = assign_quantiles(train_df[a], a, quantiles[a])
            test_df[a] = assign_quantiles(test_df[a], a, quantiles[a])
    
    ids = list(range(1,23843))
    indiv_errors_train = []
    alphas = []
    train_predictions_list = []
    test_predicitions_list = []
    tree_indiv_error = []
    
    # Build the trees, make predictions, and calculate error 
    for num in [65, 300, 1000]:
        for i in range(1,num):
            print(f"Building tree {i} for {num}")
            tree = ID3_Tree()
            build_set = attribute_set.copy()
            tree.build_tree(build_set, train_df, 'information_gain', 0, 1, test_df)
            train_predictions = tree.make_predictions(train_df)
            train_predictions_list.append(train_predictions)
            test_predicitions_list.append(tree.make_predictions(test_df))
            indiv_errors_train.append(tree.calculate_error(train_df, train_predictions))
            total_error = tree.calculate_stump_alpha(train_predictions, train_df)
            tree_indiv_error.append(total_error)
            alpha = 0.5 * log((1-total_error)/total_error)
            alphas.append(alpha)
            tree.update_weights(train_df,train_predictions, alpha)
            
    
        relevant_test_predictions = test_predicitions_list[0:(num-1)]
        overall_test_predictions = tree.make_total_predictions(relevant_test_predictions, alphas)
        
        numerics = []
    
        for x in overall_test_predictions:
            if x == 'yes':
                numerics.append(1)
            else:
                numerics.append(0)
        
        results = {'ID':ids, 'Prediction':numerics}
        result_df = pd.DataFrame.from_dict(results)
        result_df.to_csv(f"./results/ada-boost-{num}_quartiles.csv", index = False)
    
if __name__ == "__main__":
    main()