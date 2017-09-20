#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

from zope import interface

from nti.schema.field import Object

logger = __import__('logging').getLogger(__name__)


class IModel(interface.Interface):
    """
    Interface that models a generic learning model.

    Every model should be made persistent, and in the machine learning
    case we are using MySQL with pickle type.
    """

    def get_pickle():
        """
        Defines the persistent pickle for this learning model.
        """


class IDataFrame(interface.Interface):
    """
    An underlying frame to hold and manipulate data.
    """


class IDataSet(interface.Interface):
    """
    Interface that models an underlying data set
    for a learning model constructed with an DataFrame.
    """

    data = Object(IDataFrame,
                  title=u"Data Frame",
                  description=u"The main data frame for the data set",
                  default=None)

    def get_from_frame(index):
        """
        Pulls a particular row from the data frame
        """
