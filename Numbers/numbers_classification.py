from Numbers.utilities import preprocessing
from os.path import dirname, join
from scripts import nearest_neighbor
import scipy.io as sio

data_all_mat = join(dirname(__file__), 'data', 'data_all.mat')
data_all = sio.loadmat(data_all_mat, spmatrix=False, mat_dtype=True)

train = data_all['trainv']
train_labels = data_all['trainlab'].flatten()
train_dictionary = preprocessing.separate_classes(train, train_labels)
template, template_labels = preprocessing.create_templates(train_dictionary, M=64, num_classes=10)

test = data_all['testv']
test_labels = data_all['testlab'].flatten()
num_test = int(data_all['num_test'][0][0])

nearest_neighbor.nearest_neighbor_with_whole_training_set_as_template(test, test_labels, train, train_labels)
nearest_neighbor.nearest_neighbor_with_10M_templates(test, test_labels, template, template_labels)
nearest_neighbor.k_nearest_neighbor_with_10M_templates(test, test_labels, template, template_labels, k=7)