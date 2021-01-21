################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

from neuralnet import *
from main import shuffle
import math
import copy



def train(x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    return five things -
        training and validation loss and accuracies - 1D arrays of loss and accuracy values per epoch.
        best model - an instance of class NeuralNetwork. You can use copy.deepcopy(model) to save the best model.
    """
    train_acc = []
    valid_acc = []
    train_loss = []
    valid_loss = []
    best_model = None
    NUM_EPOCH = config['epochs']
    EARLY_STOP = config['early_stop']
    EARLY_STOP_EPOCH = config['early_stop_epoch']
    BATCH_SIZE = config['batch_size']
    model = NeuralNetwork(config=config)
    loss = float('inf')
    best_loss = float('inf')
    best_accuracy = 0
    patience = 0



    for i in range (NUM_EPOCH):

        x_train, y_train = shuffle(x_train, y_train)
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        for j in range (0, len(x_train), BATCH_SIZE):
            start = j
            end = j + BATCH_SIZE
            if (end > len(x_train)):
                end = len(x_train)

            x = x_train[start:end]
            y = y_train[start:end]

            model.forward(x, y) 
            model.backward()

        train_epoch_loss = model.forward(x_train, y_train)
        
        train_predict = np.zeros_like(model.y)
        train_predict[np.arange(len(model.y)), model.y.argmax(1)] = 1

        train_accuracy = sum([1 if all(train_predict[i] == y_train[i]) else 0 for i in range(len(y_train))])/len(y_train)

        train_loss.append(train_epoch_loss)
        train_acc.append(train_accuracy)
        
        valid_epoch_loss = model.forward(x_valid, y_valid)
        valid_predict = np.zeros_like(model.y)
        valid_predict[np.arange(len(model.y)), model.y.argmax(1)] = 1

        valid_accuracy = sum([1 if all(valid_predict[i] == y_valid[i]) else 0 for i in range(len(y_valid))])/len(y_valid)

        valid_loss.append(valid_epoch_loss)
        valid_acc.append(valid_accuracy)


        print("Epoch:", i, "Train Accuracy|Loss:", train_accuracy,"| ", train_epoch_loss, "~|~ Valid: ", valid_accuracy, " | ", valid_epoch_loss)
        if EARLY_STOP:
            if valid_epoch_loss > best_loss and patience >= EARLY_STOP_EPOCH:
                return train_acc, valid_acc, train_loss, valid_loss, best_model
            elif valid_epoch_loss > best_loss and patience < EARLY_STOP_EPOCH:
                patience += 1
            else:
                patience = 0
            if valid_epoch_loss < best_loss:
                best_loss = valid_epoch_loss
                best_accuracy = valid_accuracy
                best_model = copy.deepcopy(model)

        loss = valid_epoch_loss

 
    best_model = model        
    return train_acc, valid_acc, train_loss, valid_loss, best_model


def test(model, x_test, y_test):
    """
    Does a forward pass on the model and return loss and accuracy on the test set.
    """
    loss = model.forward(x_test, y_test)
    predict = np.zeros_like(model.y)
    predict[np.arange(len(model.y)), model.y.argmax(1)] = 1

    accuracy = sum([1 if all(predict[i] == y_test[i]) else 0 for i in range(len(y_test))])/len(y_test)

    return loss, accuracy
    raise NotImplementedError("Test method not implemented")
