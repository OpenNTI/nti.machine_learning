#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from zope import interface

from zope.schema import List
from zope.schema import Float

from nti.machine_learning.interfaces import IModel
from nti.machine_learning.interfaces import IDataSet

class ISupervisedModel(IModel):
    """
    Interface that models a supervised learning model
    """
    success_rate = Float(title=u"Success Rate",
                         description=u"How successful the model was at prediction on the validation set")
    
    def classify(inputs):
        """
        Classifies a set of inputs
        """
        
    def train():
        """
        Govern training of the model
        """
    
class ISupervisedDataSet(IDataSet):
    """
    Outlines the necessary components to structure
    a data set for use by a learning model. That is, 
    keeping index lists for each use within the model (training
    and validation).
    """
    
    _indices = List(title=u"Indices",
                    description=u"The main list of indices within the underlying data frame")
    
    _training_indices = List(title=u"Training Indices",
                             description=u"The list of indices of the data frame to be used for training")
    
    _validation_indices = List(title=u"Validation Indices",
                               description=u"The list of indices of the data frame to be used for validation")
    
    def get_training_set_inputs():
        """
        Fetches a list of inputs for training
        """
    
    def get_training_set_outputs():
        """
        Fetches a list of the expected outputs for training
        """
        
    def get_validation_set_inputs():
        """
        Fetches a list of the inputs for validation
        """
        
    def get_validation_set_outputs():
        """
        Fetches a list of the expected outputs for validation
        """

class INeuralNetwork(ISupervisedModel):
    """
    Outlines an ANN learning model that has an
    adaptable structure.
    """
    
class ISVM(ISupervisedModel):
    """
    Outlines a Support Vector Machine learning model.
    """