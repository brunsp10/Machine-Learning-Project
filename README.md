# Contents
Contains all code and datasets used for the Machine Learning class competitive project. Also contains all results in the results subfolder.

# How to Run
None of the files require any additional arguments. python \<file_name\> or python3 \<file_name\> will be sufficient to run them.

# Python scripts
All output in the results folder are from the below scripts.
1. ada-boost.py - runs the AdaBoosting algorithm with decision stumps
2. ada-boost_add_numerics.py - adds a column containing AdaBoost score for 300 decision stumps based on categorical variables to the test and train datasets
3. data_cleaning.py - encodes all categorical attributes with an integer for distinct values then runs SVM using an rbf kernel and the SVC method from scikit-learn's SVM package
4. gradient_descent.py - calculates optimal weight vector for LMS regression
5. neural_net_cross_validation.py - performs cross validation for gamma for 3 layer artificial neural networks
6. nn.py - runs the neural network classification algorithm
7. nn_tuning.py - used to pick the number of epochs to run the neural network classification methods
8. perceptron.py - runs the averaged perceptron to classify samples
9. perceptron_cross_validation.py - performs cross validation to pick learning rate for the averaged perceptron
10. poly_svm_ada.py - runs the sklearn.SVM.SVC method with polynomial kernel for continuous variables with an added variable containing AdaBoosted stump scores
11. random_forest.py - runs Random Forest decision tree algorithm
12. rbf_svm_ada.py - runs the sklearn.SVM.SVC method with rbf kernel for continuous variables with an added variable containing AdaBoosted stump scores
13. rbf_svm_all_attr.py - runs the sklearn.SVM.SVC method with rbf kernel for the full dataset where categorical variables have been encoded based on distinct values for each attribute
14. sklearn_SVM_poly.py - runs cross validation then sklearn.SVM.SVC method with polynomial kernel for continuous variables
15. sklearn_SVM_rbf.py - runs cross validation then sklearn.SVM.SVC method with rbf kernel for continuous variables
16. split_test.py - testing splitting with sklearn sklearn.model_selection.KFold method

# Datasets
1. test_final.csv - the test dataset downloaded from Kaggle
2. train_final.csv - the training dataset downloaded from Kaggle
3. test_final_gd.csv - test dataset containing only continuous variables
4. test_final_gd_reduced.csv - test dataset containing continuous variables except “capital.gain” and “capital.loss"
5. test_with_ada.csv - test dataset containing continuous variables and a new attribute that contains a score from 300 AdaBoosted decision stumps
6. train_0_1_label.csv - training data with the "yes" and "no" label replaced by 0 and 1 respectively
7. train_0_1_label_gd.csv - training data containing only continuous variables with the "yes" and "no" label replaced by 0 and 1 respectively
8. train_0_1_label_gd.csv - training data containing continuous variables except “capital.gain” and “capital.loss" with the "yes" and "no" label replaced by 0 and 1 respectively
9. train_with_ada.csv - training dataset containing continuous variables and a new attribute that contains a score from 300 AdaBoosted decision stumps
10. train_final_gd.csv - training dataset containing only continuous variables
11. train_final_gd_reduced.csv - training dataset containing continuous variables except “capital.gain” and “capital.loss"