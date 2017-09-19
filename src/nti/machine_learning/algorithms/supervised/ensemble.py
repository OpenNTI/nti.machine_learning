#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from zope import interface

from sklearn.ensemble import RandomForestClassifier

from nti.machine_learning.algorithms.supervised import SupervisedModel

from nti.machine_learning.algorithms.supervised.interfaces import IEnsembleRandomForestClassifier

from nti.machine_learning.evaluation.cross_validation import KFoldCrossValidation


@interface.implementer(IEnsembleRandomForestClassifier)
class EnsembleRandomForestClassifier(SupervisedModel):

    def classify(self, inputs):
        self.cls.predict(inputs)

    def train(self, data_frame, prediction_column, metric='accuracy', k=10, **kwargs):
        super(EnsembleRandomForestClassifier, self).train(data_frame, prediction_column)
        self.cls = RandomForestClassifier(**kwargs)
        kf = KFoldCrossValidation(self.cls, self._data.get_frame_no_predictor(),
                                  self._data.get_predictors(), k, metric)
        self.success_rate = kf.compute_scores().mean()
