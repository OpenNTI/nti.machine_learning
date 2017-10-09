#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

from sklearn.neural_network import MLPClassifier

from zope import interface

from nti.machine_learning.algorithms.supervised import SupervisedModel

from nti.machine_learning.algorithms.supervised.interfaces import INeuralNetwork

from nti.machine_learning.evaluation.cross_validation import KFoldCrossValidation

logger = __import__('logging').getLogger(__name__)


@interface.implementer(INeuralNetwork)
class NeuralNetwork(SupervisedModel):
    """
    Abstraction of a multi-layer perceptron classifier from sci-kit learn
    """

    def classify(self, inputs):
        return self.clf.predict([inputs])

    def train(self, data_frame, prediction_columns, metric='accuracy', k=10, **kwargs):
        super(NeuralNetwork, self).train(data_frame,
                                         prediction_columns)
        self.clf = MLPClassifier(**kwargs)
        kf = KFoldCrossValidation(self.clf, self._data.get_frame_no_predictor(),
                                  self._data.get_predictors(), k, metric)
        scores = kf.compute_scores()
        self.clf.fit(self._data.get_frame_no_predictor(), self._data.get_predictors())
        self.success_rate = scores.mean()
