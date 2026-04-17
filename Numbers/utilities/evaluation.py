from os.path import dirname, join
import matplotlib.pyplot as plt

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
    

def generating_correctly_and_misclassified_images(test, test_labels, predicted_labels):
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

    generate_number_images(misclassified, "One misclassified image per class (1-NN)", "misclassified_images_1NN.png")
    generate_number_images(correctly_classified, "One correctly classified image per class (1-NN)", "correctly_classified_images_1NN.png")
