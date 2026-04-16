import numpy as np

def nearest_neighbor_classification_with_templates(test, template, template_labels, chunk_size=1000) -> np.ndarray:
    num_test = len(test)
    predicted_labels = np.zeros(num_test)
    template_norm = np.sum(template**2, axis=1)
    
    for i in range(0, num_test, chunk_size):
        test_chunk = test[i:i+chunk_size]
        test_norm = np.sum(test_chunk**2, axis=1, keepdims=True)
        distances = test_norm + template_norm - 2 * test_chunk @ template.T
        nearest_idx = np.argmin(distances, axis=1)
        predicted_labels[i:i+chunk_size] = template_labels[nearest_idx]
    
    return predicted_labels