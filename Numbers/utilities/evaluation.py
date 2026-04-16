import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, join
import seaborn as sns
from sklearn import metrics

def generate_confusion_matrix_and_error_rates(test_labels, predicted_labels, suptitle, filename, labels=range(10)):
    confusion = metrics.confusion_matrix(test_labels, predicted_labels, labels=labels)
    error_rate = np.mean(np.array(test_labels) != np.array(predicted_labels))
    
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.suptitle(suptitle)
    plt.title(f"Error Rate: {error_rate:.4f}")
    plt.savefig(join(dirname(dirname(__file__)), 'results', filename))
    
    
def generate_number_images(images, suptitle, filename):
    figure, axes = plt.subplots(2, 5, figsize=(10,4))
    for c, ax in enumerate(axes.flat):
        if images[c] is not None:
            ax.imshow(images[c], cmap='gray')
        ax.set_title(str(c))
        ax.axis('off')

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(join(dirname(dirname(__file__)), 'results', filename))