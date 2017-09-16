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

from nti.machine_learning.model_evaluation.cross_validation import KFoldCrossValidation


@interface.implementer(INeuralNetwork)
class NeuralNetwork(SupervisedModel):
    """
    Abstraction of a multi-layer perceptron classifier from sci-kit learn
    """

    def classify(self, inputs):
        return self.clf.predict([inputs])

    def train(self, data_frame, prediction_columns, layers, **kwargs):
        super(NeuralNetwork, self).train(data_frame,
                                         prediction_columns)
        self.clf = MLPClassifier(hidden_layer_sizes=layers, **kwargs)
        kf = KFoldCrossValidation(self.clf, self._data.get_frame_no_predictor(),
                                  self._data.get_predictors(), 10, 'accuracy')
        scores = kf.compute_scores()
        self.success_rate = scores.mean()
