from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset
import numpy as np
random.seed(1215)
torch.manual_seed(1215)


def iris_data:
    iris = load_iris()

class makeData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self,index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

def _train_test_split(data, target, test_size):
    train_X, test_X, train_y, test_y = train_test_split(data, target, test_size)
    return train_X, test_X, train_y, test_y

def get_train_data(input):
    train_X, test_X, train_y, test_y = _train_test_split(input.data, input.target, test_size)
    train_data = makeData(np.array(normalize(train_X)), np.array(train_y))
    return train_data

def get_test_data(input):
    train_X, test_X, train_y, test_y = _train_test_split(input.data, input.target, test_size)
    test_data = makeData(np.array(normalize(test_X)),np.array(test_y))
    return test_data

if __name__ == "__main__":
    iris = load_iris()
    train_data = get_train_data(iris,test_size = 0.1)
    test_data = get_test_data(iris,test_size = 0.1)
