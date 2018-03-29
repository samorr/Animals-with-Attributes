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

def bz_unpickle(filename):
    return cPickle.load(bz2.BZ2File(filename))

# adapt these paths and filenames to match local installation

LOCAL_PATH = '../'
FEATURE_PATTERN = LOCAL_PATH + 'code/feat/%s-%s.pic.bz2'
LABELS_PATTERN =  LOCAL_PATH + 'code/feat/%s-labels.pic.bz2'

all_features = ['cq','lss','phog','sift','surf','rgsift']

features_length = {}

attribute_matrix = 2*np.loadtxt(LOCAL_PATH + 'predicate-matrix-binary.txt', dtype=float)-1
classnames = loadstr(LOCAL_PATH + 'classes.txt', nameonly)
attributenames = loadstr(LOCAL_PATH + 'predicates.txt', nameonly)

train_classes = loadstr(LOCAL_PATH + 'trainclasses.txt')
test_classes = loadstr(LOCAL_PATH + 'testclasses.txt')

def create_data(all_classes, attribute_id):
    """ create set of data from all statistics """
    featurehist = {}
    for feature in all_features:
        featurehist[feature] = []
    
    labels = []
    for classname in all_classes:
        for feature in all_features:
            featurefilename = FEATURE_PATTERN % (classname,feature)
            histfile = bz_unpickle(featurefilename)
            featurehist[feature].extend( histfile )
        
        labelfilename = LABELS_PATTERN % classname
        labels.extend( bz_unpickle(labelfilename)[:,attribute_id] )
    
    for feature in all_features:
        temp = np.array(featurehist[feature])
        featurehist[feature] = (temp - temp.mean(axis=0)) / np.maximum(temp.std(axis=0),1)
    
    labels = np.array(labels)

    for feature in all_features:
        features_length[feature] = featurehist[feature].shape[1]

    return featurehist,labels


def flatten_data_set(dataSet):
    data = np.concatenate([dataSet['cq'],dataSet['lss'],dataSet['phog'], dataSet['sift'], dataSet['surf'], dataSet['rgsift']], axis=1)
    return data

def collect_histograms(all_classes):
    featurehist = {}
    for feature in all_features:
        featurehist[feature] = []
    
    labels = []
    for classname in all_classes:
        for feature in all_features:
            featurefilename = FEATURE_PATTERN % (classname,feature)
            histfile = bz_unpickle(featurefilename)
            featurehist[feature].extend( histfile )

        labelfilename = LABELS_PATTERN % classname
        labels.extend( bz_unpickle(labelfilename) )
    
    labels = (np.array(labels) + 1.) / 2.
    for feature in all_features:
        temp = np.array(featurehist[feature])
        featurehist[feature] = (temp - temp.mean(axis=0)) / np.maximum(temp.std(axis=0),1)

    for feature in all_features:
        features_length[feature] = featurehist[feature].shape[1]

    return flatten_data_set(featurehist), labels

