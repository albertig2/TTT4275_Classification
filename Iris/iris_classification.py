from scripts import model
import numpy as np


class_features = np.loadtxt("Iris/data/iris.data", delimiter=",", usecols=(0, 1, 2, 3)).reshape(3, 50, 4)
N_training = 30; N_testing = 20

training_set_first_30 = class_features[:, :N_training, :].reshape(-1, 4)
testing_set_last_20 = class_features[:, N_training:, :].reshape(-1, 4)
X_training = np.column_stack((training_set_first_30, np.ones(3*N_training))).T
X_testing = np.column_stack((testing_set_last_20, np.ones(3*N_testing))).T

X_training_last_30 = np.column_stack((class_features[:, N_testing:, :].reshape(-1, 4), np.ones(3*N_training))).T
X_testing_first_20 = np.column_stack((class_features[:, :N_testing, :].reshape(-1, 4), np.ones(3*N_testing))).T

target_training = np.zeros((3, 3*N_training)); target_training[0, :N_training] = 1; target_training[1, N_training:2*N_training] = 1; target_training[2, 2*N_training:] = 1 
test_labels = np.zeros(N_testing*3); test_labels[0:N_testing] = 0; test_labels[N_testing:2*N_testing] = 1; test_labels[2*N_testing:3*N_testing] = 2

model.plot_feature_histograms(class_features, "seperability_histograms", 8)
  

features_combinations = [range(0,5), range(1, 5), range(2, 5), range(3, 5)]

#Train and evaluate ALL the models that uses the first 30 samples as a training set
model.train_and_evaluate(X_training, X_testing, features_combinations, target_training, test_labels, "Iris/results/default_training_set")

#Train and evaluate the model that uses the last 30 samples as a training set
model.train_and_evaluate(X_training_last_30, X_testing_first_20, [features_combinations[0]], target_training, test_labels, "Iris/results/last_30_training_set")

