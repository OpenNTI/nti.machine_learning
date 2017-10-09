#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from six.moves import cPickle

from pandas import Series
from pandas import DataFrame

from zope import interface

from nti.machine_learning.interfaces import IModel
from nti.machine_learning.interfaces import IDataSet

FORMAT = '%(asctime)-15s %(message)s'

logger = __import__('logging').getLogger(__name__)


@interface.implementer(IModel)
class Model(object):
    """
    An abstract model for a data task.
    """

    def get_pickle(self):
        """
        Get the pickled model for persistent storage
        """
        return cPickle.dumps(self)


@interface.implementer(IDataSet)
class AbstractDataSet(object):
    """
    User for the management of data frames
    while algorithms execute
    """

    def get_from_frame(self, index):
        """
        Get a row from the core data frame at index "index"
        """
        try:
            # Try to get it by key
            row = self._data.loc[index].as_matrix()
            answer = self._prediction_data.loc[index].as_matrix()
        except:
            try:
                # If that doesn't work, get by numeric index
                row = self._data.iloc[index].as_matrix()
                answer = self._prediction_data.iloc[index].as_matrix()
            except:
                msg = "Index %s could not be found in data set."
                raise ValueError(msg % (index,))
        return (row, answer)
    _get_from_frame = get_from_frame  # BWC
