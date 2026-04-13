from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = dirname(__file__)
mat_fname = pjoin(data_dir, 'data_all.mat')

mat_contents = sio.loadmat(mat_fname, spmatrix=False, mat_dtype=True)

test = mat_contents['testv']

# Task 2.1 Nearest-Neighborhood Classifier

# Task 2.1.1a

template = mat_contents['trainv']
template_labels = mat_contents['trainlab'].flatten()
num_template = int(mat_contents['num_train'][0][0])

test = mat_contents['testv']
test_labels = mat_contents['testlab'].flatten()
num_test = int(mat_contents['num_test'][0][0])

labels = [0,1,2,3,4,5,6,7,8,9]
"""
def euclidean_distance(x, ref):
    return (x - ref) @ (x - ref)

y_pred = np.zeros_like(testlab)

for t in range(num_test):
    min_distance = float('inf')
    closest_class = 0
    for d in range(num_train):
        distance = euclidean_distance(testv[t,:], trainv[d,:])
        if distance < min_distance:
            min_distance = distance
            closest_class = trainlab[d]
    y_pred[t] = closest_class
"""   
    
         
# Compute squared Euclidean distances between all test and train samples

chunk_size = 1000
y_pred = np.zeros(num_test)

for i in range(0, num_test, chunk_size):
    test_chunk = test[i:i+chunk_size]
    dists = (
        np.sum(test_chunk**2, axis=1, keepdims=True)
        + np.sum(template**2, axis=1)
        - 2 * test_chunk @ template.T
    )

    # For each test sample, find the nearest training sample
    nearest_idx = np.argmin(dists, axis=1)

    # Predict labels
    y_pred[i:i+chunk_size] = template_labels[nearest_idx]

confusion = confusion_matrix(test_labels, y_pred, labels=labels)


plt.figure(figsize=(8,6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for Nearest-Neighbor Classifier")
plt.show()

error_rate = np.mean(np.array(test_labels) != np.array(y_pred))
print(f"Error Rate: {error_rate:.4f}")


# Task 2.1.1b
num_classes = 10
misclassified = [None] * num_classes
correctly_classified = [None] * num_classes
for i in range(num_test):
    true_label = int(test_labels[i])
    pred_label = int(y_pred[i])
    
    if pred_label != true_label and misclassified[true_label] is None:
        misclassified[true_label] = test[i,:].reshape((28,28))
        
    if pred_label == true_label and correctly_classified[true_label] is None: 
        correctly_classified[true_label] = test[i,:].reshape((28,28))

    # Stop early if we have one for every class
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

plot_number_images(misclassified, "One misclassified example per class (1-NN)")
plot_number_images(correctly_classified, "One correctly classified example per class (1-NN)")

#x = test[i,:].reshape((28, 28)) # Converts the image vector (number i) to a 28x28 matrix
#plt.imshow(x, cmap='gray') # Plots the matrix x
#plt.show() 
"""
distance = np.linalg.norm(template - test) # Calculates the Euclidean distance between two vectors. For multiple vectors.
# For multiple vectors you can use: 
from scipy.spatial.distance import cdist
cdist(template, test, metric='euclidean')
"""