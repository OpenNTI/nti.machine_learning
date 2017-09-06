#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from sklearn.neural_network import MLPClassifier

from nti.machine_learning.algorithms.supervised import SupervisedModel


class NeuralNetwork(SupervisedModel):
    """
    Abstraction of a multi-layer perceptron classifier 
    from sci-kit learn
    """

    def __init__(self, data_frame, prediction_column, layers, training_size=.7, **kwargs):
        super(NeuralNetwork, self).__init__(data_frame,
                                            prediction_column, 
                                            training_set_ratio=training_size)
        self.mlp = MLPClassifier(hidden_layer_sizes=layers, **kwargs)

    def classify(self, inputs):
        return self.mlp.predict([inputs])

    def train(self):
        self.mlp.fit(self._training_set_inputs, self._training_set_outputs)
        self._run_validation()
