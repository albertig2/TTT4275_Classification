from common_utilities.evaluation import generate_confusion_matrix_and_error_rates
from Numbers.utilities import classification, evaluation
from sklearn import neighbors
import numpy as np

def nearest_neighbor_with_whole_training_set_as_template(test, test_labels, template, template_labels) -> np.ndarray:  
    predicted_labels = classification.nearest_neighbor_classification_with_templates(test, template, template_labels, chunk_size=1000)
    generate_confusion_matrix_and_error_rates(test_labels, predicted_labels, "Confusion Matrix for Nearest-Neighbor Classifier the whole training set as template", "Numbers/results/confusion_matrix_whole_training_set.png", range(10))
    return predicted_labels
   
   
def nearest_neighbor_with_10M_templates(test, test_labels, template, template_labels) -> np.ndarray:
    predicted_labels = classification.nearest_neighbor_classification_with_templates(test, template, template_labels, chunk_size=1000)  
    generate_confusion_matrix_and_error_rates(test_labels, predicted_labels, "Confusion Matrix for Nearest-Neighbor Classifier with 10*64 templates", "Numbers/results/confusion_matrix_10M_templates.png", range(10))    
    return predicted_labels


def k_nearest_neighbor_with_10M_templates(test, test_labels, template, template_labels, k=7) -> np.ndarray:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k,  metric='euclidean')
    knn.fit(template, template_labels)

    predicted_labels = knn.predict(test)
    generate_confusion_matrix_and_error_rates(test_labels, predicted_labels, "Confusion Matrix for KNN Classifier with 10*64 templates (K=7)", "Numbers/results/confusion_matrix_KNN_10M_templates.png", range(10))
    return predicted_labels