import numpy as np
from sklearn.model_selection import train_test_split, KFold

def train_val_test_split(x, y, val_ratio, test_ratio, rand_seed=None):
    print('Splitting into train, validation, test...')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=rand_seed)
    x_valid, y_valid = np.array([]), np.array([])
    if val_ratio > 0.0:
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, 
            test_size=val_ratio / (1. - test_ratio), stratify=y_train, random_state=rand_seed)

    print('Train images/labels:', x_train.shape, y_train.shape)
    print('Validation images/labels:', x_valid.shape, y_valid.shape)
    print('Test images/labels:', x_test.shape, y_test.shape)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def cross_validation_split(x, y, num_folds, rand_seed=None):
    kf = KFold(n_splits=num_folds, random_state=rand_seed, shuffle=True)
    for train_index, test_index in kf.split(x):
        yield x[train_index], y[train_index], x[test_index], y[test_index]