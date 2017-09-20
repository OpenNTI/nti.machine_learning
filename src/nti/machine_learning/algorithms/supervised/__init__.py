#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

from zope import interface

from nti.machine_learning import Model
from nti.machine_learning import DataFrame
from nti.machine_learning import AbstractDataSet

from nti.machine_learning.algorithms.supervised.interfaces import ISupervisedModel
from nti.machine_learning.algorithms.supervised.interfaces import ISupervisedDataSet

from nti.property.property import alias

from nti.schema.fieldproperty import createDirectFieldProperties

from nti.schema.schema import SchemaConfigured

logger = __import__('logging').getLogger(__name__)


@interface.implementer(ISupervisedDataSet)
class SupervisedDataSet(AbstractDataSet, SchemaConfigured):
    """
    Class managing a data set for use by
    a supervised learning model.
    """
    createDirectFieldProperties(ISupervisedDataSet)

    data = alias('_data')
    prediction_column = alias('_prediction_column')

    def __init__(self, data_frame, prediction_column):
        self._data = data_frame
        self._prediction_column = prediction_column
        try:
            self._prediction_data = self._data[prediction_column]
            self._data = self._data.drop(prediction_column, axis=1)
        except IndexError:
            raise ValueError("Invalid prediction column.")

    def total_size(self):
        """
        Get the total size of the data set
        """
        return len(self._data.index)

    def get_frame_no_predictor(self):
        """
        Gets a matrix of input values without the predictor
        """
        matrix = self._data.as_matrix()
        return matrix

    def get_predictors(self):
        """
        Gets a matrix of only the predictors
        """
        matrix = self._prediction_data.as_matrix()
        if len(matrix.shape) > len(self._prediction_column):
            shape = matrix.shape
            matrix = matrix.reshape(shape[:-1])
        return matrix


@interface.implementer(ISupervisedModel)
class SupervisedModel(Model, SchemaConfigured):
    """
    A supervised learning model
    """
    createDirectFieldProperties(ISupervisedModel)

    data = alias('_data')

    def classify(self, inputs):
        """
        Classify a set of inputs
        """
        raise NotImplementedError("classify function not implemented")

    def train(self, data_frame, prediction_columns):
        """
        Train the model. Time consuming, therefore it is its own method
        that must be called rather than in the constructor.
        """
        if not isinstance(data_frame, DataFrame):
            raise TypeError("data_frame must be of type DataFrame")
        if len(data_frame) <= 1:
            raise ValueError("Insufficient data set size")
        self._data = SupervisedDataSet(data_frame, prediction_columns)
