#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from sklearn.neural_network import MLPClassifier

from zope import interface

from nti.machine_learning.algorithms.supervised import SupervisedModel

from nti.machine_learning.algorithms.supervised.interfaces import INeuralNetwork

from nti.machine_learning.evaluation.cross_validation import KFoldCrossValidation


@interface.implementer(INeuralNetwork)
class NeuralNetwork(SupervisedModel):
    """
    Abstraction of a multi-layer perceptron classifier from sci-kit learn
    """

    def __init__(self, data_frame, prediction_column, layers, **kwargs):
        super(NeuralNetwork, self).__init__(data_frame,
                                            prediction_column)
        self.classifier = MLPClassifier(hidden_layer_sizes=layers, **kwargs)

    def classify(self, inputs):
        return self.classifier.predict([inputs])

    def train(self):
        kf = KFoldCrossValidation(self.classifier, self._data.get_frame_no_predictor(),
                                  self._data.get_predictors(), 10, 'accuracy')
        scores = kf.compute_scores()
        self.success_rate = scores.mean()
