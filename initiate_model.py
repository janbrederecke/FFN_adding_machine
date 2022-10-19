"""
This script contains a function to create an instance of our FFN, the
optimizer, and the loss function
"""

import torch
import torch.nn as nn
from FFNAddingMachine import FFNAddingMachine


def initiate_model(depth=1, width=32, dropout=0.0, batch_normalize=False, learning_rate=0.001):

    # create model instance, loss function, and optimizer
    model_instance = FFNAddingMachine(depth=depth,
                                      width=width,
                                      dropout=dropout,
                                      batch_normalize=batch_normalize)

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(model_instance.parameters(), lr=learning_rate)

    return model_instance, loss_function, optimizer
