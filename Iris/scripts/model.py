from common_utilities.evaluation import generate_confusion_matrix_and_error_rates
import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, join


def MSE(G, T):
    """
    Input: the model's estimate G and the wanted target T
    Output: The MSE scalar
    """
    MSE = 0
    for col in range(np.shape(G)[1]):
        MSE += (G[:,col] - T[:,col]).T @ (G[:,col] - T[:,col])
    return 1/2 * MSE

def grad_MSE(G, T, X):
    """
    Input: the model's estimate G, the wanted target T and the training data X
    Output: the gradient of MSE with respect to W.
    """
    delta = (G - T) * G * (1 - G)
    return delta @ X.T 

def train_model(X, T):
    """
    Trains the model W based on the input data X and the solution provided by T.
    Output: Trained matrix W
    """
    DIM = np.shape(X)[0] - 1; C = np.shape(T)[0]
    W = np.zeros((C, DIM + 1)) 
    G = 1 / (1 + np.exp(-W @ X))

    alpha = 0.001
    tresh = 10**(-6)
    prev_err = 1000000; curr_err = 0.9*prev_err
    rel_err = (prev_err - curr_err) / curr_err

    while (rel_err > tresh):
        W = W - alpha * grad_MSE(G, T, X)
        Z = W @ X
        G = 1 / (1 + np.exp(-Z))
        prev_err = curr_err
        curr_err = MSE(G, T)
        rel_err = (prev_err - curr_err) / curr_err
    return W

def get_predicted_classes(X, W):
    """
    Gets the predicted class for a given classifier W and input data X. 
    """
    Z = W @ X
    G = 1 / (1 + np.exp(-Z))
    C = np.argmax(G, axis=0) 
    return C

def train_and_evaluate(X_train, X_test, feature_subsets, target_training, test_labels, results_path):
    DIM = np.shape(X_train)[0] - 1
    for i, features in enumerate(feature_subsets):
        W = train_model(X_train[features, :], target_training)
        predicted_labels = get_predicted_classes(X_test[features, :], W)
        n_features = DIM - i
        C = np.shape(target_training)[0]
        generate_confusion_matrix_and_error_rates(
            test_labels, predicted_labels,
            f"Classifier performance using {n_features} features",
            f"{results_path}/confusion_matrix_{n_features}_features.png",
            range(C)
        )

def plot_feature_histograms(class_features, folder, BINS = 10):
    """
    Plots the histograms for each feature of a C x N x D input.
    """
    DIM = np.shape(class_features)[2]
    C = np.shape(class_features)[0]

    for i in range(DIM):
        for j in range(C):
            plt.hist(class_features[j, :, i], alpha=0.8, label="Class" + str(j), bins=BINS)
        plt.title("Feature " + str(i))
        plt.legend()
        plt.savefig(join(dirname(dirname(__file__)), 'results', folder, "Histogram_feature_" + str(i)))
        plt.cla()
