"""
This script contains a function to create data for the adding machine    
"""

import numpy as np

def create_data(size):
    """
    This function creates data for the feed forward neural network that will
    be trained as an adding machine
    """

    # train_data
    train_data = np.array([np.random.randint(-10, 10, size),
                            np.random.randint(-10, 10, size)]).T
    row_sum = np.sum(train_data, axis = 1)[np.newaxis,:].T
    train_data = np.hstack((train_data, row_sum))

    # test_data
    test_data = np.array([np.random.randint(-10, 10, int(size * 0.5)),
                            np.random.randint(-10, 10, int(size * 0.5))]).T
    row_sum = np.sum(test_data, axis = 1)[np.newaxis,:].T
    test_data = np.hstack((test_data, row_sum))

    # holdout_data
    holdout_data = np.array([np.random.randint(-10, 10, int(size * 0.5)),
                            np.random.randint(-10, 10, int(size * 0.5))]).T
    row_sum = np.sum(holdout_data, axis = 1)[np.newaxis,:].T
    holdout_data = np.hstack((holdout_data, row_sum))

    return train_data, test_data, holdout_data
