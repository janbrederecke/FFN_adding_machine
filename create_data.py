"""
This script contains a function to create data for the adding machine    
"""

import numpy as np

def create_data(size):
    """
    This function creates data for the feed forward neural network that will
    be trained as an adding machine
    """

    train_data = np.random.randint(-10, 10, size)
    test_data = np.random.randint(-10, 10, int(0.5 * size))
    holdout_data = np.random.randint(-10, 10, int(0.5 * size))

    return train_data, test_data, holdout_data
