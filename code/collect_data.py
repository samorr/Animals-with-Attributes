#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:45:36 2018

Using
Animals with Attributes Dataset, http://attributes.kyb.tuebingen.mpg.de
(C) 2009 Christoph Lampert <chl@tuebingen.mpg.de>

@author: dominik
"""

import numpy as np

import _pickle as cPickle
import bz2

def nameonly(x):
    return x.split('\t')[1]

def loadstr(filename, converter=str):
    return [converter(c.strip()) for c in open(filename).readlines()]

def bzUnpickle(filename):
    return cPickle.load(bz2.BZ2File(filename))

# adapt these paths and filenames to match local installation

# local_path = '/home/dominik/Dokumenty/Studia/Data-mining/Animals-with-Attributes/'
local_path = '../'
feature_pattern = local_path + 'code/feat/%s-%s.pic.bz2'
labels_pattern =  local_path + 'code/feat/%s-labels.pic.bz2'

all_features = ['cq','lss','phog','sift','surf','rgsift']

features_length = {}

attribute_matrix = 2*np.loadtxt(local_path + 'predicate-matrix-binary.txt', dtype=float)-1
classnames = loadstr(local_path + 'classes.txt', nameonly)
attributenames = loadstr(local_path + 'predicates.txt', nameonly)

train_classes = loadstr(local_path + 'trainclasses.txt')
test_classes = loadstr(local_path + 'testclasses.txt')

def createData(all_classes, attribute_id):
    """ create set of data from all statistics """
    featurehist = {}
    for feature in all_features:
        featurehist[feature] = []
    
    labels = []
    for classname in all_classes:
        for feature in all_features:
            featurefilename = feature_pattern % (classname,feature)
            print('# ',featurefilename)
            histfile = bzUnpickle(featurefilename)
            featurehist[feature].extend( histfile )
        
        labelfilename = labels_pattern % classname
        print('# ',labelfilename)
        print('#')
        labels.extend( bzUnpickle(labelfilename)[:,attribute_id] )
    
    for feature in all_features:
        temp = np.array(featurehist[feature])
        featurehist[feature] = (temp - temp.mean(axis=0)) / np.maximum(temp.std(axis=0),1)
    
    labels = np.array(labels)

    for feature in all_features:
        features_length[feature] = featurehist[feature].shape[1]

    return featurehist,labels


def flattenDataSet(dataSet):
    data = np.concatenate([dataSet['cq'],dataSet['lss'],dataSet['phog'], dataSet['sift'], dataSet['surf'], dataSet['rgsift']], axis=1)
    return data

def collectHistograms(all_classes):
    featurehist = {}
    for feature in all_features:
        featurehist[feature] = []
    
    labels = []
    for classname in all_classes:
        for feature in all_features:
            featurefilename = feature_pattern % (classname,feature)
            print('# ',featurefilename)
            histfile = bzUnpickle(featurefilename)
            featurehist[feature].extend( histfile )

        labelfilename = labels_pattern % classname
        print('# ',labelfilename)
        print('#')
        labels.extend( bzUnpickle(labelfilename) )
    
    labels = (np.array(labels) + 1.) / 2.
    for feature in all_features:
        temp = np.array(featurehist[feature])
        featurehist[feature] = (temp - temp.mean(axis=0)) / np.maximum(temp.std(axis=0),1)

    for feature in all_features:
        features_length[feature] = featurehist[feature].shape[1]

    return flattenDataSet(featurehist), labels

