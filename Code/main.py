################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################
# We've provided you with the dataset in PA2.zip
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

from utils import load_data, load_config, write_to_file, one_hot_encoding
from train import *
import random
from matplotlib import pyplot as plt


def shuffle(X, Y):
    assert len(X) == len(Y)
    
    joined = list(zip(X, Y))
    random.shuffle(joined)
    return zip(*joined)

def normalize(X, mean, std):
    return (X - mean) / std
    
def one_hot_encode(num, max_num=10):    
    encoding = np.zeros((num.size, num.max() + 1))
    encoding[np.arange(num.size), num] = 1
    return encoding


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./config.yaml")

    # Load the data
    x_train, y_train, x_test, y_test = load_data()

    # one hot encode the y values
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)
    
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    x_train = x_train.reshape(len(x_train), x_train.shape[1] * x_train.shape[2])
    x_test = x_test.reshape(len(x_test), x_test.shape[1] * x_test.shape[2])



    # Create validation set out of training data.
    x_train_data = x_train[:int((len(x_train)+1)*.80)]
    y_train_data = y_train[:int((len(y_train)+1)*.80)]

    x_val = x_train[int(len(x_train)*.80+1):]
    y_val = y_train[int(len(y_train)*.80+1):]

    # Any pre-processing on the datasets goes here.
    train_mean = np.mean(x_train_data, axis = 0)
    train_std = np.std(x_train_data, axis = 0)


    x_train_data = normalize(x_train_data, train_mean, train_std)
    x_val = normalize(x_val, train_mean, train_std)
    x_test = normalize(x_test, train_mean, train_std)

    # train the model
    train_acc, valid_acc, train_loss, valid_loss, best_model = \
        train(x_train_data, y_train_data, x_val, y_val, config)

    test_loss, test_acc = test(best_model, x_test, y_test)

    print("Config: %r" % config)
    print("Test Loss", test_loss)
    print("Test Accuracy", test_acc)


    plt.errorbar(range(len(train_loss)), train_loss, color = 'blue', label='train loss')
    plt.errorbar(range(len(valid_loss)), valid_loss, color = 'red', label='valid loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title('Loss For Each Epoch')

    
    plt.legend()
    plt.show()

    plt.errorbar(range(len(train_loss)), train_acc, color = 'blue', label='train accuracy')
    plt.errorbar(range(len(valid_acc)), valid_acc, color = 'red', label='valid accuracy')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy For Each Epoch')

    
    plt.legend()
    plt.show()

    # DO NOT modify the code below.
    data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
            'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}



    write_to_file('./results.pkl', data)
