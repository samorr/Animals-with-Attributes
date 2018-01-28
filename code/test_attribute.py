#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 09:03:03 2018

Using
Animals with Attributes Dataset, http://attributes.kyb.tuebingen.mpg.de
(C) 2009 Christoph Lampert <chl@tuebingen.mpg.de>

@author: dominik
"""


import numpy as numpy
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import collect_data as cd

inputSVM_pattern =  './results/%s-SVM.pic.bz2'
inputLogReg_pattern =  './results/%s-LogReg.pic.bz2'


def testLogRegAccuracy(attributeId):
    dataSet, labels = cd.createData(cd.test_classes, attributeId)
    data = cd.flattenDataSet(dataSet)

    filename = inputLogReg_pattern % cd.attributenames[attributeId]
    logreg = cd.bzUnpickle(filename)

    print('Score for logistic regression for attribute {1}:\n{2}', cd.attributenames[attributeId], logreg.score(data, labels))

