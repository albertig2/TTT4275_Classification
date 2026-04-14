from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plotting_confusion_matrix_and_error_rates(test_labels, predicted_labels, suptitle, labels=range(10)):
    confusion = metrics.confusion_matrix(test_labels, predicted_labels, labels=labels)
    error_rate = np.mean(np.array(test_labels) != np.array(predicted_labels))
    
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.suptitle(suptitle)
    plt.title(f"Error Rate: {error_rate:.4f}")
    plt.show()