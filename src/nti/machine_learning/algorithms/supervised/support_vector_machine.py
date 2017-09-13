#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from sklearn.svm import SVC

from zope import interface

from nti.machine_learning.algorithms.supervised import SupervisedModel

from nti.machine_learning.algorithms.supervised.interfaces import ISVM

from nti.machine_learning.algorithms.supervised.interfaces import DEFAULT_TRAINING_SIZE


@interface.implementer(ISVM)
class SupportVectorMachine(SupervisedModel):
    """
    Abstraction of the SciKit Learn Support Vector Machine.
    """

    def __init__(self, data_frame, prediction_column, 
                 training_size=DEFAULT_TRAINING_SIZE, **kwargs):
        super(SupportVectorMachine, self).__init__(data_frame,
                                                   prediction_column,
                                                   training_set_ratio=training_size)
        self.clf = SVC(**kwargs)

    def classify(self, inputs):
        return self.clf.predict([inputs])

    def train(self):
        self.clf.fit(self._training_set_inputs, self._training_set_outputs)
        self.success_rate = self.clf.score(self._validation_set_inputs, 
                                           self._validation_set_outputs)
