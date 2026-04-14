from os.path import dirname, join
import scipy.io as sio
from scripts import nearest_neighbor

data_all_mat = join(dirname(__file__), 'data', 'data_all.mat')
data_all = sio.loadmat(data_all_mat, spmatrix=False, mat_dtype=True)

train = data_all['trainv']
train_labels = data_all['trainlab'].flatten()

test = data_all['testv']
test_labels = data_all['testlab'].flatten()
num_test = int(data_all['num_test'][0][0])

nearest_neighbor.nearest_neighbor_with_whole_training_set_as_template(test, test_labels, train, train_labels)
nearest_neighbor.nearest_neighbor_with_10M_templates(test, test_labels, train, train_labels)
nearest_neighbor.k_nearest_neighbor_with_10M_templates(test, test_labels, train, train_labels, k=7)