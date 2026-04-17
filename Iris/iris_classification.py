from common_utilities.evaluation import generate_confusion_matrix_and_error_rates
from Iris.scripts import model_training, model_testing
from Iris.utilities import evaluation
import matplotlib.pyplot as plt
import numpy as np



# All data about each feature of each class

class_features = np.loadtxt("Iris/data/iris.data", delimiter=",", usecols=(0, 1, 2, 3)).reshape(3, 50, 4)
N_training = 30; N_testing = 20

training_set_first_30 = class_features[:, :N_training, :].reshape(-1, 4)
testing_set_last_20 = class_features[:, N_training:, :].reshape(-1, 4)
X_training = np.column_stack((training_set_first_30, np.ones(3*N_training))).T
X_testing = np.column_stack((testing_set_last_20, np.ones(3*N_testing))).T

#For 1d) spefically
testing_set_first_20 = class_features[:, :N_testing, :].reshape(-1, 4)
training_set_last_30 = class_features[:, N_testing:, :].reshape(-1, 4)
X_training_last_30 = np.column_stack((training_set_last_30, np.ones(3*N_training))).T
X_testing_first_20 = np.column_stack((testing_set_first_20, np.ones(3*N_testing))).T

#Define the correct target solution
target_training = np.zeros((3, 3*N_training)); target_training[0, :N_training] = 1; target_training[1, N_training:2*N_training] = 1; target_training[2, 2*N_training:] = 1 
test_labels = np.zeros(N_testing*3); test_labels[0:N_testing] = 0; test_labels[N_testing:2*N_testing] = 1; test_labels[2*N_testing:3*N_testing] = 2


#Plot histograms to figure out which features to use for next step
dimention = 4; classes = 3
evaluation.generate_histograms_for_fetures(class_features, dimention, classes)



def train_and_evaluate(X_train, X_test, feature_subsets, test_labels, results_path):
    for i, features in enumerate(feature_subsets):
        W = model_training.train_model(X_train[features, :], target_training)
        predicted_labels = model_testing.get_predicted_classes(X_test[features, :], W)
        n_features = 4 - i
        generate_confusion_matrix_and_error_rates(
            test_labels, predicted_labels,
            f"Classifier performance using {n_features} features",
            f"{results_path}/confusion_matrix_{n_features}_features.png",
            range(classes)
        )    

features_combinations = [range(0,5), range(1, 5), range(2, 5), range(3, 5)] #iteration i uses i + 1 features. By coincidence the worst feature to use is always the one with the lowest index, so we can easily write the features like this....
#Train ALL the models that uses the first 30 samples as a training set. Then quantify their performance.
train_and_evaluate(X_training, X_testing, features_combinations, test_labels, "../Iris/results/default_training_set")
#Train the model that uses the last 30 samples as a training set, and quantify performance..
train_and_evaluate(X_training_last_30, X_testing_first_20, [features_combinations[0]], test_labels, "../Iris/results/last_30_training_set")

