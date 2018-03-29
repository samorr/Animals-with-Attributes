#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 09:03:03 2018

Using
Animals with Attributes Dataset, http://attributes.kyb.tuebingen.mpg.de
(C) 2009 Christoph Lampert <chl@tuebingen.mpg.de>

@author: dominik
"""


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import collect_data as cd

INPUT_SVM_PATTERN =  './results/%s-SVM.pic.bz2'
INPUT_LOG_REG_PATTERN =  './results/%s-LogReg-C=%s.pic.bz2'
INPUT_LOG_REG_PATTERN_SIMPLE =  './results/%s-LogReg.pic.bz2'
INPUT_LOG_REG_CV_PATTERN =  './results/%s-LogRegCV.pic.bz2'

def test_all_log_reg_accuracy():
    scores = []
    data, labels = cd.collect_histograms(cd.test_classes)
    labels = 2 * labels -1
    for i in range(85):
        scores.append(test_classifier(data, labels[:,i], i, INPUT_LOG_REG_PATTERN_SIMPLE % cd.attributenames[i]))
    return np.array(scores)

def test_all_log_reg_CV_accuracy():
    scores = []
    data, labels = cd.collect_histograms(cd.test_classes)
    labels = 2 * labels -1
    for i in range(85):
        scores.append(test_classifier(data, labels[:,i], i, INPUT_LOG_REG_CV_PATTERN % cd.attributenames[i]))
    return np.array(scores)

def test_log_reg_accuracy(attribute_id, C):
    data_set, labels = cd.create_data(cd.test_classes, attribute_id)
    data = cd.flatten_data_set(data_set)
    filename = INPUT_LOG_REG_PATTERN % (cd.attributenames[attribute_id], str(C))

    return test_classifier(data, labels, attribute_id, filename)

def test_classifier(data, labels, attribute_id, filename):
    classifier = cd.bz_unpickle(filename)
    score = classifier.score(data, labels)
    print('Score for attribute {0}:\n{1}'.format(cd.attributenames[attribute_id], score))
    return score

def test_log_reg_CV_accuracy(attribute_id):
    data_set, labels = cd.create_data(cd.test_classes, attribute_id)
    data = cd.flatten_data_set(data_set)
    filename = INPUT_LOG_REG_CV_PATTERN % cd.attributenames[attribute_id]

    return test_classifier(data, labels, attribute_id, filename)

def score_classifiers(classes, input_pattern):
    attribute_matrix = (cd.attribute_matrix + 1.) / 2.
    attributes_prob = np.sum(attribute_matrix, axis=0) / 50
    class_prob = np.prod(np.abs(1 - attributes_prob[np.newaxis,:] - attribute_matrix), axis=1)
    class_prob = np.array([class_prob[i] for i in range(50) if cd.classnames[i] in classes])
    attribute_matrix = np.array([attribute_matrix[i,:] for i in range(50) if cd.classnames[i] in classes]).astype(np.int)

    data, labels = cd.collect_histograms(classes)
    classifiers_prob = []
    for attribute_id in range(85):
        filename = input_pattern % cd.attributenames[attribute_id]
        classifier = cd.bz_unpickle(filename)
        classifiers_prob.append(classifier.predict_proba(data))
    
    classifiers_prob = np.array(classifiers_prob)
    err = 0
    predicted = []

    for test_sample in range(data.shape[0]):
        predicted_class = np.argmax(np.prod(classifiers_prob[ :,np.newaxis, test_sample, 0] *(1-attribute_matrix.T) + classifiers_prob[ :, np.newaxis,test_sample, 1] * attribute_matrix.T, axis=0) / class_prob)

        predicted.append(predicted_class)
        if np.any(attribute_matrix[predicted_class,:] != labels[test_sample,:]):
            err += 1
    err /= data.shape[0]
    print('Total average score: ', 1. - err)
    predicted = np.array(predicted)
    return predicted, get_confusion_matrix(predicted, labels, attribute_matrix)

def get_confusion_matrix(predicted, labels, attribute_matrix):
    def which_class(attributes):
        return [i for i in range(attribute_matrix.shape[0]) if np.all(attribute_matrix[i,:] == attributes)][0]

    confusion_matrix = np.zeros((attribute_matrix.shape[0], attribute_matrix.shape[0]))
    for i in range(len(predicted)):
        confusion_matrix[which_class(attribute_matrix[predicted[i]]), which_class(labels[i,:])] += 1
    return confusion_matrix.astype(np.int)