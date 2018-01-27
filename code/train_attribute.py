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
from sklearn.metrics.pairwise import chi2_kernel
import collect_data as cd

import _pickle as cPickle
import bz2

def bzPickle(obj,filename):
    f = bz2.BZ2File(filename, 'wb')
    cPickle.dump(obj, f)
    f.close()

outputSVM_pattern =  './results/%s-SVM.pic.bz2'

gamma = np.ones(6)  

def statKernel(x1, x2):
    cqKernel = chi2_kernel(x1[:cd.features_length['cq']], x2[:cd.features_length['cq']], gamma=gamma[0])

    lssKernel = chi2_kernel(x1[cd.features_length['cq']:cd.features_length['lss']], x2[cd.features_length['cq']:cd.features_length['lss']], gamma=gamma[1])

    phogKernel = chi2_kernel(x1[cd.features_length['lss']:cd.features_length['phog']], x2[cd.features_length['lss']:cd.features_length['phog']], gamma=gamma[2])

    siftKernel = chi2_kernel(x1[cd.features_length['phog']:cd.features_length['sift']], x2[cd.features_length['phog']:cd.features_length['sift']], gamma=gamma[3])

    surfKernel = chi2_kernel(x1[cd.features_length['sift']:cd.features_length['surf']], x2[cd.features_length['sift']:cd.features_length['surf']], gamma=gamma[4])

    rgsiftKernel = chi2_kernel(x1[cd.features_length['surf']:cd.features_length['rgsift']], x2[cd.features_length['surf']:cd.features_length['rgsift']], gamma=gamma[5])

    kernels = np.array([cqKernel, lssKernel, phogKernel, siftKernel, surfKernel, rgsiftKernel])

    return np.sum(kernels)

def trainAttribute(attributeId):
    data, labels = cd.createData(cd.train_classes, attributeId)
    trainData = cd.flattenDataSet(data)
    
    svm = SVC(C=10., kernel='rbf', probability=True)
    svm.fit(trainData, labels)

    filename = outputSVM_pattern % cd.attributenames[attributeId]
    bzPickle(svm, filename)
    
