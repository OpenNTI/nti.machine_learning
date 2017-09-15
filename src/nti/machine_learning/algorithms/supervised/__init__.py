#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from numpy import array

from numpy.random import shuffle

from zope import interface

from nti.machine_learning import Model
from nti.machine_learning import DataFrame
from nti.machine_learning import AbstractDataSet

from nti.machine_learning.algorithms.supervised.interfaces import ISVM
from nti.machine_learning.algorithms.supervised.interfaces import INeuralNetwork
from nti.machine_learning.algorithms.supervised.interfaces import ISupervisedModel
from nti.machine_learning.algorithms.supervised.interfaces import ISupervisedDataSet

from nti.property.property import alias

from nti.schema.fieldproperty import createDirectFieldProperties

from nti.schema.schema import SchemaConfigured

DEFAULT_TRAINING_SET_RATIO = 0.7


@interface.implementer(ISupervisedDataSet)
class SupervisedDataSet(AbstractDataSet, SchemaConfigured):
    """
    Class managing a data set for use by
    a supervised learning model.
    """
    createDirectFieldProperties(ISupervisedDataSet)

    data = alias('_data')
    indices = alias('_indices')
    training_ratio = alias('_training_ratio')
    prediction_column = alias('_prediction_column')

    def __init__(self, data_frame, prediction_column, training_ratio):
        self._data = data_frame
        self._training_ratio = training_ratio
        self._prediction_column = prediction_column
        try:
            self._prediction_data = self._data[prediction_column]
            self._data = self._data.drop(prediction_column, axis=1)
        except IndexError:
            raise ValueError("Invalid prediction column.")
        self._indices = list(data_frame.index.values)
        shuffle(self._indices)
        training_size = int(len(self._indices) * self._training_ratio)
        self._training_indices = [
            self._indices[i] for i in range(training_size)
        ]
        self._validation_indices = [
            self._indices[i] for i in self._indices if i not in self._training_indices
        ]

    def get_training_set_inputs(self):
        """
        Get the inputs for the training set
        """
        return [self.get_from_frame(i)[0] for i in self._training_indices]

    def get_training_set_outputs(self):
        """
        Get the outputs for the training set
        """
        return [self.get_from_frame(i)[1] for i in self._training_indices]

    def get_validation_set_inputs(self):
        """
        Get the inputs for a validation set
        """
        return [self.get_from_frame(i)[0] for i in self._validation_indices]

    def get_validation_set_outputs(self):
        """
        Get the outputs for the validation
        """
        return [self.get_from_frame(i)[1] for i in self._validation_indices]

    def total_size(self):
        """
        Get the total size of the data set
        """
        return len(self._data.index)

    def get_frame_no_predictor(self):
        """
        Gets a matrix of input values without the predictor
        """
        return self._data.as_matrix()

    def get_predictors(self):
        """
        Gets a matrix of only the predictors
        """
        return self._prediction_data.as_matrix()


@interface.implementer(ISupervisedModel)
class SupervisedModel(Model, SchemaConfigured):
    """
    A supervised learning model
    """
    createDirectFieldProperties(ISupervisedModel)

    data = alias('_data')
    training_set_inputs = alias('_training_set_inputs')
    training_set_outputs = alias('_training_set_outputs')
    validation_set_inputs = alias('_validation_set_inputs')
    validation_set_outputs = alias('_validation_set_outputs')

    def __init__(self, data_frame, prediction_column, training_set_ratio=DEFAULT_TRAINING_SET_RATIO):
        if not isinstance(data_frame, DataFrame):
            raise TypeError("data_frame must be of type DataFrame")
        if len(data_frame) <= 1:
            raise ValueError("Insufficient data set size")
        self._data = SupervisedDataSet(data_frame, prediction_column,
                                       training_ratio=training_set_ratio)
        self._training_set_inputs = self._data.get_training_set_inputs()
        self._training_set_outputs = self._data.get_training_set_outputs()
        self._validation_set_inputs = self._data.get_validation_set_inputs()
        self._validation_set_outputs = self._data.get_validation_set_outputs()

    def classify(self, inputs):
        """
        Classify a set of inputs
        """
        raise NotImplementedError("classify function not implemented")

    def train(self):
        """
        Train the model. Time consuming, therefore it is its own method
        that must be called rather than in the constructor.
        """
        raise NotImplementedError("train function not implemented")
