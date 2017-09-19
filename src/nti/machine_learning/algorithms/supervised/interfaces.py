#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from nti.machine_learning.interfaces import IModel
from nti.machine_learning.interfaces import IDataSet

from nti.schema.field import Number


class ISupervisedModel(IModel):
    """
    Interface that models a supervised learning model
    """
    success_rate = Number(title=u"Success Rate",
                          description=u"How successful the model was at prediction on the validation set",
                          required=False)

    def classify(inputs):
        """
        Classifies a set of inputs

        inputs:
            The inputs to be classified.
        """

    def train(data_frame, prediction_columns):
        """
        Govern training of the model

        data_frame:
            The DataFrame containing all necessary data
            to train the model.

        prediction_columns:
             A list of columns the model is attempting
             to predict.
        """


class ISupervisedDataSet(IDataSet):
    """
    Outlines the necessary components to structure
    a data set for use by a learning model. That is,
    manage the structure of the data for training and
    validation
    """

    def total_size():
        """
        Return: The total size of the data set
        """

    def get_frame_no_predictor():
        """
        Return: A subset of the original data frame
        that contains only the features of the data set
        """

    def get_predictors():
        """
        Return: A subset of the original data frame
        that contains the predictors of the data set
        in a shape compliant to KFoldCrossValidation
        """


class INeuralNetwork(ISupervisedModel):
    """
    Outlines an ANN learning model that has an
    adaptable structure.
    """

    def train(data_frame, prediction_columns, layers):
        """
        Trains the Articial Neural Netowork

        The only difference between this train and the
        ISupervisedModel train is this train allows for the
        configurability of the layers, which is required rather
        than using a default in kwargs.
        """


class ISVM(ISupervisedModel):
    """
    Outlines a Support Vector Machine learning model.
    """

class ILinearSupportVectorClassification(ISupervisedModel):
    """
    Outlines a Linear Support Vector Classification
    """


class IRegressor(ISupervisedModel):
    """
    Outlines a regression model
    """


class IKNearestNeighborsClassifier(ISupervisedModel):
    """
    Outlines of a k-Nearest Neighbors classifier
    """


class IKNearestNeighborsRegressor(ISupervisedModel):
    """
    Outlines of a k-Nearest Neighbors regressor
    """

class IEnsembleRandomForestClassifier(ISupervisedModel):
    """
    Outlines of a Random Forest Classifier
    """
