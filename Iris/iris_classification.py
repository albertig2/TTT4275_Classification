from common_utilities.evaluation import generate_confusion_matrix_and_error_rates
from Iris.scripts import model_training, model_testing
from Iris.utilities import evaluation
import matplotlib.pyplot as plt
import numpy as np



# All data about each feature of each class
class1_features = np.loadtxt("Iris/data/class_1", delimiter=",")
class2_features = np.loadtxt("Iris/data/class_2", delimiter=",")
class3_features = np.loadtxt("Iris/data/class_3", delimiter=",")

#Seperate into training and test data
N_training = 30; N_testing = 20
class1_training = class1_features[:N_training]; class1_testing = class1_features[N_training:] 
class2_training = class2_features[:N_training]; class2_testing = class2_features[N_training:]
class3_training = class3_features[:N_training]; class3_testing = class3_features[N_training:]

# Define the extended X that includes unit row vector, for each classes dataset.
X1_all_features_training = np.column_stack((class1_training, np.ones(N_training))).T; X1_all_features_testing = np.column_stack((class1_testing, np.ones(N_testing))).T
X2_all_features_training = np.column_stack((class2_training, np.ones(N_training))).T; X2_all_features_testing = np.column_stack((class2_testing, np.ones(N_testing))).T
X3_all_features_training = np.column_stack((class3_training, np.ones(N_training))).T; X3_all_features_testing = np.column_stack((class3_testing, np.ones(N_testing))).T

#Combine into one training set 
X_all_features_training = np.concatenate((X1_all_features_training, X2_all_features_training, X3_all_features_training), axis=1) 
X_all_features_testing = np.concatenate((X1_all_features_testing, X2_all_features_testing, X3_all_features_testing), axis=1)

C = 3 #Amount of classes
#Define the correct target solution
Target_all_features_training = np.zeros((3, 3*N_training)); Target_all_features_training[0, :N_training] = 1; Target_all_features_training[1, N_training:2*N_training] = 1; Target_all_features_training[2, 2*N_training:] = 1 
Target_all_features_testing = np.zeros((3, 3*N_testing)); Target_all_features_testing[0, :N_testing] = 1; Target_all_features_testing[1, N_testing:2*N_testing] = 1; Target_all_features_testing[2, 2*N_testing:] = 1 
Test_labels = np.zeros(N_testing*3); Test_labels[0:N_testing] = 0; Test_labels[N_testing:2*N_testing] = 1; Test_labels[2*N_testing:3*N_testing] = 2

#Plot histograms to figure out which features to use for next step
evaluation.generate_histograms_for_fetures(class1_features, class2_features, class3_features, 4)

#Train the models
#Model using all 4 features
W_all_features = model_training.train_model(X_all_features_training, Target_all_features_training)
#Model using 3 features; Skipping the 0th feature as that seems to be the last valuable one.
W_three_features = model_training.train_model(X_all_features_training[[1, 2, 3, 4], :], Target_all_features_training)
#Model using 2 features; skipping the 0th and 1th feature as they seem to have most overlapping probability distributions.
W_two_features = model_training.train_model(X_all_features_training[[2, 3, 4], :], Target_all_features_training)
#Model using 1 feature; the 3rd feature seems to have the least overlap, so we choose this one...
W_one_features = model_training.train_model(X_all_features_training[[3, 4], :], Target_all_features_training)

W_matrices = [W_all_features, W_three_features, W_two_features, W_one_features]

for i, W in enumerate(W_matrices):
    predicted_labels = model_testing.get_predicted_classes(X_all_features_testing[range(i, 5), :], W)
    generate_confusion_matrix_and_error_rates(Test_labels, predicted_labels, "Classifier performance, " + str(4 - i) + " features",f"../Iris/results/confusion_matrix_{4 -i}_features.png", range(3) )


