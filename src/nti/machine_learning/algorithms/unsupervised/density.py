#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from sklearn.cluster import DBSCAN as SK_DBSCAN

from zope import interface

from nti.machine_learning.algorithms.unsupervised import AbstractClusterModel

from nti.machine_learning.algorithms.unsupervised.interfaces import IDBScan


@interface.implementer(IDBScan)
class DBScan(AbstractClusterModel):
    """
    Performs the DBSCAN clustering algorithm using
    a Pandas-provided data frame.

    This algorithm does a walk through of the data,
    splitting clusters when the eps paramter is not met
    """

    def cluster(self, data_frame, eps=0.1, min_samples=50, **kwargs):
        super(DBScan, self).cluster(data_frame)
        self.cls = SK_DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        return self.cls.fit_predict(self._data.to_matrix())

    @property
    def eps(self):
        return self.cls.eps

    @property
    def min_samples(self):
        return self.cls.min_samples
