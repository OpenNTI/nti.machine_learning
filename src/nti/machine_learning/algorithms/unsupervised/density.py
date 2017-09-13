#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from sklearn.cluster import DBSCAN

from zope import interface

from nti.machine_learning.algorithms.unsupervised import AbstractClusterModel

from nti.machine_learning.algorithms.unsupervised.interfaces import IDBScan

@interface.implementer(IDBScan)
class DBScan(AbstractClusterModel):

    def __init__(self, data_frame, **kwargs):
        super(DBScan, self).__init__(data_frame)
        self.cls = DBSCAN(**kwargs)

    def cluster(self):
        return self.cls.fit_predict(self._data.to_matrix())