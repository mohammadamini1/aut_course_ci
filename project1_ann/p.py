#!/usr/bin/python3

import random
import pickle
import math
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=50)



def loading_datasets(shuffle_sets=True):
    # loading training set features
    f = open("ANN_Project_Assets/Datasets/train_set_features.pkl", "rb")
    train_set_features2 = pickle.load(f)
    f.close()

    # reducing feature vector length
    features_STDs = np.std(a=train_set_features2, axis=0)
    train_set_features = train_set_features2[:, features_STDs > 52.3]

    # changing the range of data between 0 and 1
    train_set_features = np.divide(train_set_features, train_set_features.max())

    # loading training set labels
    f = open("ANN_Project_Assets/Datasets/train_set_labels.pkl", "rb")
    train_set_labels = pickle.load(f)
    f.close()

    # ------------
    # loading test set features
    f = open("ANN_Project_Assets/Datasets/test_set_features.pkl", "rb")
    test_set_features2 = pickle.load(f)
    f.close()

    # reducing feature vector length
    features_STDs = np.std(a=test_set_features2, axis=0)
    test_set_features = test_set_features2[:, features_STDs > 48]

    # changing the range of data between 0 and 1
    test_set_features = np.divide(test_set_features, test_set_features.max())

    # loading test set labels
    f = open("ANN_Project_Assets/Datasets/test_set_labels.pkl", "rb")
    test_set_labels = pickle.load(f)
    f.close()

    # ------------
    # preparing our training and test sets - joining datasets and lables
    train_set = []
    test_set = []

    for i in range(len(train_set_features)):
        label = np.array([0,0,0,0])
        label[int(train_set_labels[i])] = 1
        label = label.reshape(4,1)
        train_set.append((train_set_features[i].reshape(102,1), label))

    for i in range(len(test_set_features)):
        label = np.array([0,0,0,0])
        label[int(test_set_labels[i])] = 1
        label = label.reshape(4,1)
        test_set.append((test_set_features[i].reshape(102,1), label))

    # shuffle
    if shuffle_sets:
        random.shuffle(train_set)
        random.shuffle(test_set)

    return train_set, test_set

def prepare_matrix(m):
    min = np.min(m) * -1
    m = m + min
    max = np.max(m)
    m = m / max
    return m



def sigmoid_activation(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
	return x * (1.0 - x)

def sigmoid_deriv_matrix(x):
    sigmoid_deriv_v = np.vectorize(sigmoid_deriv)
    return sigmoid_deriv_v(x)



def calculate(a, w, b):
    mul = np.matmul(a, w)
    res = np.add(mul, b)
    sigmoid_activation_v = np.vectorize(sigmoid_activation)
    return sigmoid_activation_v(res)

def calculate2(a, w, b):
    mul = np.matmul(a, w)
    res = np.add(mul, b)
    sigmoid_activation_v = np.vectorize(sigmoid_activation)
    return res, sigmoid_activation_v(res)



def step2_feedforwarding():
    train_set, test_set = loading_datasets()
    train_set = train_set[:200]

    w_102_150 = prepare_matrix(np.random.normal(0,1, (102, 150)))
    w_150_60  = prepare_matrix(np.random.normal(0,1, (150, 60)))
    w_60_4    = prepare_matrix(np.random.normal(0,1, (60, 4)))
    b_150     = np.zeros((1, 150))
    b_60      = np.zeros((1, 60))
    b_4       = np.zeros((1, 4))

    hit_counts = 0
    for image in train_set:
        r = calculate(image[0].T, w_102_150, b_150)
        r = calculate(r, w_150_60, b_60)
        r = calculate(r, w_60_4, b_4)

        image_true_result = image[1].tolist().index([1])
        if image_true_result == r.argmax():
            hit_counts += 1

    hit_rate = hit_counts / train_set.__len__() * 100
    return hit_rate


def step4_vectorization():
    train_set, test_set = loading_datasets()
    train_set = train_set[:200]
    # test_set = test_set[:50]

    w_102_150 = prepare_matrix(np.random.normal(0,1, (102, 150))).astype('float64')
    w_150_60  = prepare_matrix(np.random.normal(0,1, (150, 60))).astype('float64')
    w_60_4    = prepare_matrix(np.random.normal(0,1, (60, 4))).astype('float64')
    b_150     = np.zeros((1, 150))
    b_60      = np.zeros((1, 60))
    b_4       = np.zeros((1, 4))

    learning_rate = 1
    number_of_epochs = 10
    batch_size = 10
    costs = []
    cost_averages = []
    for epoch in range(number_of_epochs):
        random.shuffle(train_set)
        batches = [
            train_set[x : x + batch_size] for x in range(0, train_set.__len__(), batch_size)
        ]
        new_costs = []
        for batch in batches:
            grad_w_102_150 = np.zeros((102, 150))
            grad_w_150_60  = np.zeros((150, 60))
            grad_w_60_4    = np.zeros((60, 4))

            grad_b_150     = np.zeros((1, 150))
            grad_b_60      = np.zeros((1, 60))
            grad_b_4       = np.zeros((1, 4))

            for image in batch:
                x = image[0]
                y = image[1]

                ## compute the output
                z1, a1 = calculate2(x.T, w_102_150, b_150)
                z2, a2 = calculate2(z1, w_150_60, b_60)
                z3, a3 = calculate2(z2, w_60_4, b_4)
                z3 = z3.T
                a3 = a3.T

                grad_w_60_4 += np.matmul((2 * sigmoid_deriv_matrix(z3) * (a3 - y)), a2).T
                grad_b_4 += np.sum(2 * sigmoid_deriv_matrix(z3) * (a3 - y), axis=1, keepdims=True).T
                grad_a2 = np.matmul(w_60_4, (2 * sigmoid_deriv_matrix(z3) * (a3 - y))).T

                grad_w_150_60 += np.matmul((2 * sigmoid_deriv_matrix(z2) * (a2 - grad_a2)).T, a1).T
                grad_b_60 += np.sum(2 * sigmoid_deriv_matrix(z2) * (a2 - grad_a2), axis=1, keepdims=True)
                grad_a1 = np.matmul(w_150_60, (2 * sigmoid_deriv_matrix(z2) * (a2 - grad_a2)).T).T

                grad_w_102_150 += np.matmul((2 * sigmoid_deriv_matrix(z1) * (a1 - grad_a1)).T, x.T).T
                grad_b_150 += np.sum(2 * sigmoid_deriv_matrix(z1) * (a1 - grad_a1), axis=1, keepdims=True)

                ## calculate costs
                c3 = np.sum(np.square((a3 - y)))
                c2 = np.sum(np.square((a2 - grad_a2)))
                c1 = np.sum(np.square((a1 - grad_a1)))
                costs.append(c1 + c2 + c3)
                new_costs.append(c1 + c2 + c3)

            ## learn
            w_102_150 = w_102_150 - (learning_rate * (grad_w_102_150 / batch_size))
            w_150_60  = w_150_60  - (learning_rate * (grad_w_150_60 / batch_size))
            w_60_4    = w_60_4    - (learning_rate * (grad_w_60_4 / batch_size))
            b_150     = b_150     - (learning_rate * (grad_b_150 / batch_size))
            b_60      = b_60      - (learning_rate * (grad_b_60 / batch_size))
            b_4       = b_4       - (learning_rate * (grad_b_4 / batch_size))

        cost_averages.append(sum(new_costs) / new_costs.__len__())
        new_costs = []

    hit_counts = 0
    for image in train_set:
        r = calculate(image[0].T, w_102_150, b_150)
        r = calculate(r, w_150_60, b_60)
        r = calculate(r, w_60_4, b_4)

        image_true_result = image[1].tolist().index([1])
        if image_true_result == r.argmax():
            hit_counts += 1

    hit_rate = hit_counts / train_set.__len__() * 100
    return hit_rate






step2_hitrate = step2_feedforwarding()
print("step2_feedforwarding rate: {:.1f} %".format(step2_hitrate,))

step4_result = 0
print("step 3:")
for t in range(10):
    res = step4_vectorization()
    step4_result += res
    print("result {}: {:.1f} %".format(t+1, res))
print("average result: {:.1f} %".format(step4_result / 10))




