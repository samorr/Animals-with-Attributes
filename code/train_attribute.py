#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:53:36 2018

Using
Animals with Attributes Dataset, http://attributes.kyb.tuebingen.mpg.de
(C) 2009 Christoph Lampert <chl@tuebingen.mpg.de>

@author: dominik
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics.pairwise import chi2_kernel
import collect_data as cd

import _pickle as cPickle
import bz2

def bz_pickle(obj,filename):
    f = bz2.BZ2File(filename, 'wb')
    cPickle.dump(obj, f)
    f.close()

OUTPUT_SVM_PATTERN =  './results/%s-SVM.pic.bz2'
OUTPUT_LOG_REG_PATTERN =  './results/%s-LogReg-C=%s.pic.bz2'
OUTPUT_LOG_REG_CV_PATTERN =  './results/%s-LogRegCV.pic.bz2'

def train_SVM(attribute_id):
    data, labels = cd.create_data(cd.train_classes, attribute_id)
    train_data = cd.flatten_data_set(data)
    
    svm = SVC(C=10., kernel='rbf', probability=True)
    svm.fit(train_data, labels)

    filename = OUTPUT_SVM_PATTERN % cd.attributenames[attribute_id]
    bz_pickle(svm, filename)
    
def train_logistic_regression(attribute_id, C):
    data, labels = cd.create_data(cd.train_classes, attribute_id)
    train_data = cd.flatten_data_set(data)
    
    logreg = LogisticRegression('l2', C=C, solver='saga')
    logreg.fit(train_data, labels)

    filename = OUTPUT_LOG_REG_PATTERN % (cd.attributenames[attribute_id], str(C))
    bz_pickle(logreg, filename)

def train_logistic_regression_CV(attribute_id):
    data, labels = cd.create_data(cd.train_classes, attribute_id)
    train_data = cd.flatten_data_set(data)
    
    logreg = LogisticRegressionCV(Cs=[0.01, 0.1, 1., 10., 100.], cv=5, dual=False, penalty='l2', solver='saga', refit=True)
    logreg.fit(train_data, labels)

    filename = OUTPUT_LOG_REG_CV_PATTERN % cd.attributenames[attribute_id]
    bz_pickle(logreg, filename)
