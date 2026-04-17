from common_utilities.evaluation import generate_confusion_matrix_and_error_rates
from sklearn import neighbors
from utilities import classification, evaluation, preprocessing

def nearest_neighbor_with_whole_training_set_as_template(test, test_labels, template, template_labels):  
    predicted_labels = classification.nearest_neighbor_classification_with_templates(test, template, template_labels, chunk_size=1000)
    generate_confusion_matrix_and_error_rates(test_labels, predicted_labels, "Confusion Matrix for Nearest-Neighbor Classifier the whole training set as template", "../Numbers/results/confusion_matrix_whole_training_set.png", range(10))
    
    num_test = len(test)
    num_classes = 10
    misclassified = [None] * num_classes
    correctly_classified = [None] * num_classes
    for i in range(num_test):
        true_label = int(test_labels[i])
        predicted_label = int(predicted_labels[i])
        
        if predicted_label != true_label and misclassified[true_label] is None:
            misclassified[true_label] = test[i,:].reshape((28,28))
            
        if predicted_label == true_label and correctly_classified[true_label] is None: 
            correctly_classified[true_label] = test[i,:].reshape((28,28))

        if all(img is not None for img in misclassified) and all(img is not None for img in correctly_classified):
            break

    evaluation.generate_number_images(misclassified, "One misclassified image per class (1-NN)", "misclassified_images_1NN.png")
    evaluation.generate_number_images(correctly_classified, "One correctly classified image per class (1-NN)", "correctly_classified_images_1NN.png")

   
def nearest_neighbor_with_10M_templates(test, test_labels, train, train_labels):
    train_dictionary = preprocessing.separate_classes(train, train_labels)
    template, template_labels = preprocessing.create_templates(train_dictionary, M=64, num_classes=10)
    predicted_labels = classification.nearest_neighbor_classification_with_templates(test, template, template_labels, chunk_size=1000)  
    generate_confusion_matrix_and_error_rates(test_labels, predicted_labels, "Confusion Matrix for Nearest-Neighbor Classifier with 10*64 templates", "../Numbers/results/confusion_matrix_10M_templates.png", range(10))    


def k_nearest_neighbor_with_10M_templates(test, test_labels, train, train_labels, k=7):
    train_dictionary = preprocessing.separate_classes(train, train_labels)
    template, template_labels = preprocessing.create_templates(train_dictionary, M=64, num_classes=10)

    knn = neighbors.KNeighborsClassifier(n_neighbors=k,  metric='euclidean')
    knn.fit(template, template_labels)

    predicted_labels = knn.predict(test)
    generate_confusion_matrix_and_error_rates(test_labels, predicted_labels, "Confusion Matrix for KNN Classifier with 10*64 templates (K=7)", "../Numbers/results/confusion_matrix_KNN_10M_templates.png", range(10))
