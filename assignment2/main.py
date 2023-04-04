######################################
# Assignement 2 for CSC420
# MNIST clasification example
# Author: Jun Gao
######################################

from read_data import *
import numpy as np
import random
from mlp import OneLayerNN, TwoLayerNN, cross_entropy_loss_function
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, train = False, batch_size = 16, shuffle = True):
        self.train = train
        self.batch_size = batch_size
        self.shuffle = shuffle

        if self.train:
            self.image = get_train_images()
            self.label = get_train_labels()
        else:
            self.image = get_test_images()
            self.label = get_test_labels()
        print('==> read data size: %d'%(len(self.image)))
        self.sequence_id = list(range(len(self.image)))
        self.cnt = 0
        if self.shuffle:
            random.shuffle(self.sequence_id)

    def __len__(self):
        return len(self.image)

    def get_one_batch(self):
        image_list = []
        label_list = []
        for i_batch in range(self.batch_size):
            if self.cnt >= len(self.image): break
            image_list.append(self.image[self.sequence_id[self.cnt]])
            label_list.append(self.label[self.sequence_id[self.cnt]])
            self.cnt += 1
        image_list = np.asarray(image_list)
        label_list = np.asarray(label_list)
        return image_list, label_list

    def reset_one_epoch(self):
        if self.train:
            random.shuffle(self.sequence_id)
        self.cnt = 0

class my_log():
    def __init__(self, file_name):
        self.log = open(file_name, 'w')

    def write(self, str_to_write):
        print(str_to_write)
        self.log.write(str_to_write)

    def flush(self):
        self.log.flush()

class Trainer():
    def __init__(self):
        self.batch_size = 64
        self.train_loader = DataLoader(train=True, shuffle=True, batch_size=self.batch_size)
        self.test_loader = DataLoader(train=False, shuffle=False, batch_size=self.batch_size)
        num_hidden_unit = 50  # number units of the hidden layer (if having two layers)
        num_input_unit = 784  # number of the input vector dimention, MNIST is 28 * 28 res, so 784 input unit
        num_out_unit = 10  # number of the output layer
        self.nn = OneLayerNN(num_input_unit, num_out_unit)
        # self.nn = TwoLayerNN(num_input_unit, num_hidden_unit, num_out_unit)
        self.learning_rate = 1e-1
        self.log = my_log('log_one_layer_nn')

    def inference(self, data_loader):
        n_correct = 0
        n_all = len(data_loader)
        n_step = len(data_loader) // self.batch_size
        for i_step in range(n_step):
            image, label = data_loader.get_one_batch()
            prediction = self.nn.forward(image)
            predict_label = np.argmax(prediction, axis=1)
            n_correct += (predict_label == label).sum()
        return n_correct / float(n_all)

    def train(self):
        train_acc, test_acc = [], []
        best_train_acc, best_test_acc = 0, 0
        for epoch in range(100):  # training epochs
            n_step = len(self.train_loader) // self.batch_size
            self.train_loader.reset_one_epoch()
            for i_step in range(n_step):
                image, label = self.train_loader.get_one_batch()
                prediction = self.nn.forward(image)
                loss = cross_entropy_loss_function(prediction, label)
                self.nn.backpropagation_with_gradient_descent(loss, self.learning_rate, image, label)

            if epoch % 1 == 0:
                self.log.write('---------\n')
                self.log.write('#epoch %d\n'%(epoch))
                self.train_loader.reset_one_epoch()
                train_correctness = self.inference(self.train_loader)
                train_acc.append(train_correctness)
                self.test_loader.reset_one_epoch()
                test_correctness = self.inference(self.test_loader)
                test_acc.append(test_correctness)
                self.log.write('train correctness %.5f'%(train_correctness))
                self.log.write('test correctness %.5f'%(test_correctness))
                if train_correctness > best_train_acc:
                    best_train_acc = train_correctness
                if test_correctness > best_test_acc:
                    best_test_acc = test_correctness

            self.log.flush()
        # Dump your trained model here.
        import pickle
        pickle.dump(self.nn, open('trained_model.pkl', 'wb'))

        print(best_train_acc, best_test_acc)
        # plot training acc
        plt.title("Learning Curve: Training Accuracy")
        plt.plot([i for i in range(100)], train_acc, label="Train")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.show()
        
        # plot testing acc
        plt.title("Learning Curve: Test Accuracy")
        plt.plot([i for i in range(100)], test_acc, label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.show()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()