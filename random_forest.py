# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 12:49:37 2023

@author: bruns
"""

import pandas as pd
from math import log2
import sys
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
    def build_tree(self, attributes, data, purity_measure, depth, max_depth, full_data, set_size):
        # If all attributes have been used or max depth reached then
        # return the most common label
        if len(attributes) == 0 or depth == max_depth:
            mode_label = data.mode()['label'][0]
            return Node(mode_label)
        
        # If all labels are the same then return that label
        elif len(data['label'].unique()) == 1:
            return Node(data['label'].unique()[0])
        
        # Otherwise, recursively build the tree
        else:
            
            # Pick the best attribute for splitting the data
            best_attribute = None
            best_gain = -1
            calculate_current = Purity_Measures(data['label'])
            current_purity = calculate_current.calculate_purity(purity_measure)
            
            # Get a random subset
            attribute_list = []
            if len(attributes) >= set_size:
                attribute_list = list(np.random.choice(list(attributes), set_size, replace = False))
            else:
                attribute_list = list(attributes)
            
            # Loop over the attributes and calculate best purity gain
            for a in attribute_list:
                single_attribute_data = data[[a,'label']]
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
                                                           full_data,
                                                           set_size)
            
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
        
        # Get label counts and total to calculate proportions
        label_counts = other_data.value_counts()
        total_items = len(other_data)
        
        # Get the labels and loop over adding entropy for
        # each label
        for label in set(other_data):
            label_items = label_counts[label]
            proportion = label_items/total_items
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
                prop = len(reduced_df)/len(self.data)
                attribute_entropy += prop * self.calculate_entropy(reduced_df['label'])
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
    
    # Replace numerical values with above/below the media
    medians = {}
    for a in attribute_list:
        if train_df[a].dtype == 'int64':
            medians[a] = train_df[a].median()
    for a in attribute_list:
        if train_df[a].dtype == 'int64':
            train_df[a] = train_df[a].apply(lambda x: x > medians[a])
            test_df[a] = test_df[a].apply(lambda x: x > medians[a])
    
    for size in [5]:
        test_predictions = []
        
        # Build the trees all at once and put them into a list 
        for i in range(1,751):
            print(f"Building tree number {i} for {size} attributes")
            tree = ID3_Tree()
            build_set = attribute_set.copy()
            random_df = train_df.sample(23842, replace = True)
            tree.build_tree(build_set, random_df, 'information_gain', 0, 18, test_df, size)
            test_predictions.append(tree.make_predictions(test_df))
        
        # Get the prediction for each value of T and check the error
        for i in range(750, 751):
            relevant_predictions_test = test_predictions[0:i]
            predicted_labels_test = []
            for j in range(len(test_df)):
                indiv_predicted_test = []
                for y in relevant_predictions_test:
                    print(j)
                    indiv_predicted_test.append(y[j])
                predicted_labels_test.append(max(set(indiv_predicted_test),
                                              key = indiv_predicted_test.count))
            
            ids = list(range(1,23843))
            
            numerics = []

            for x in predicted_labels_test:
                if x == 'yes':
                    numerics.append(1)
                else:
                    numerics.append(0)
            
            results = {'ID':ids, 'Prediction':numerics}
            result_df = pd.DataFrame.from_dict(results)
            result_df.to_csv("./results/random_forest_quartiles.csv", index = False)

if __name__ == "__main__":
    main()