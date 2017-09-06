#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

try:
    import cPickle as pickle
except ImportError:
    import pickle

from pandas import Series
from pandas import DataFrame

FORMAT = '%(asctime)-15s %(message)s'

# For right now I'm not sure if we'll need to extend
# these, but I'll leave them here just in case
NTISeries = Series
NTIDataFrame = DataFrame


class Model():
    """
    An abstract model for a data task.
    """

    def get_pickle(self):
        """
        Get the pickled model for persistent storage
        """
        return pickle.dumps(self)


class AbstractDataSet():
    """
    User for the management of data frames
    while algorithms execute
    """

    def _get_from_frame(self, index):
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
