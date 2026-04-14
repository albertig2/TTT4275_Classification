import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, join
import scipy.io as sio
import seaborn as sns
from sklearn import metrics

data_all_mat = join(dirname(dirname(__file__)), 'data', 'data_all.mat')
data_all = sio.loadmat(data_all_mat, spmatrix=False, mat_dtype=True)

# Task 2.1 Nearest-Neighborhood Classifier
## Task 2.1.1 Nearest-Neighborhood Classifier with 60000 templates
### Task 2.1.1a
template = data_all['trainv']
template_labels = data_all['trainlab'].flatten()

test = data_all['testv']
test_labels = data_all['testlab'].flatten()
num_test = int(data_all['num_test'][0][0])

#### Classifying the test samples based on the closest template sample
chunk_size = 1000
predicted_labels = np.zeros(num_test)

for i in range(0, num_test, chunk_size):
    test_chunk = test[i:i+chunk_size]
    distances = (
        np.sum(test_chunk**2, axis=1, keepdims=True)
        + np.sum(template**2, axis=1)
        - 2 * test_chunk @ template.T
    )
    nearest_idx = np.argmin(distances, axis=1)
    predicted_labels[i:i+chunk_size] = template_labels[nearest_idx]

#### Confusion matrix and error rates
labels = [0,1,2,3,4,5,6,7,8,9]

confusion = metrics.confusion_matrix(test_labels, predicted_labels, labels=labels)
plt.figure(figsize=(8,6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for Nearest-Neighbor Classifier with 60000 templates")
plt.show()

error_rate = np.mean(np.array(test_labels) != np.array(predicted_labels))
print(f"Error Rate: {error_rate:.4f}")


### Task 2.1.1b and 2.1.1c
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

    #### Stop early if we have one for every class
    if all(img is not None for img in misclassified) and all(img is not None for img in correctly_classified):
        break

def plot_number_images(images, suptitle):
    figure, axes = plt.subplots(2, 5, figsize=(10,4))
    for c, ax in enumerate(axes.flat):
        if images[c] is not None:
            ax.imshow(images[c], cmap='gray')
        ax.set_title(str(c))
        ax.axis('off')

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

plot_number_images(misclassified, "One misclassified image per class (1-NN)")
plot_number_images(correctly_classified, "One correctly classified image per class (1-NN)")
