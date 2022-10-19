"""
This script contains a function to train an instance of our FFN
"""

import torch 
from initiate_model import initiate_model


# define training function
def train_model(batch_normalize = False, dropout = 0.0, epochs = 10, width = 32, depth = 1, learning_rate = 0.001):
    
    # set number of epochs
    epochs = epochs
    print(f"Going to train {epochs} epochs.")

    # initiate model instane, lossfunction, and optimizer
    model_instance, loss_function, optimizer = initiate_model(dropout = dropout,
                                                              width = width,
                                                              depth = depth,
                                                              learning_rate = learning_rate)

    # initialize losses
    losses = torch.zeros(epochs)
    train_accuracy  = []
    test_accuracy   = []

    for epoch in range(epochs):

        if epoch == 0:
            print(f"Training epoch {epoch}")
        elif epoch % 2 == 0 and epoch + 1 != epochs:
            print(f"Training epoch {epoch},\
            last trainloss was {losses[epoch-1]},\
            last trainaccuracy was {train_accuracy[epoch-1]},\
            last testaccuracy was {test_accuracy[epoch-1]}")

        # switch on training mode
        model_instance.train()
        
        # loop over training data batches
        batch_accuracy  = []
        batch_loss      = []
        
        for X,y in train_loader:

            # forward pass and loss
            y_hat = model_instance(X)
            loss = loss_function(y_hat, y)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss from this batch
            batch_loss.append(loss.item())

            # compute accuracy
            matches = torch.argmax(y_hat, axis=1) == y     # booleans (false/true)
            matchesNumeric = matches.float()             # convert to numbers (0/1)
            accuracyPct = 100*torch.mean(matchesNumeric) # average and x100
            batch_accuracy.append( accuracyPct )               # add to list of accuracies
            # end of batch loop...

        # now that we've trained through the batches, get their average training accuracy
        train_accuracy.append( np.mean(batch_accuracy) )

        # and get average losses across the batches
        losses[epoch] = np.mean(batch_loss)

        # test accuracy
        model_instance.eval()
        X,y = next(iter(test_loader)) # extract X,y from test dataloader
        with torch.no_grad(): # deactivates autograd
            y_hat = model_instance(X)
        
        # compare the following really long line of code to the training accuracy lines
        test_accuracy.append( 100*torch.mean((torch.argmax(y_hat, axis=1)==y).float()) )
        
        # print final loss when training is done
        if epoch + 1 == epochs:
            print(f"Training finished after {epochs} epochs, last mean loss was {losses[epoch]}")

    # end epochs
    
    # function output
    return train_accuracy, test_accuracy, losses, model_instance