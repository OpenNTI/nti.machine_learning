#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from sklearn.cluster import KMeans as SK_KMeans

from zope import interface

from nti.machine_learning.algorithms.unsupervised import AbstractClusterModel

from nti.machine_learning.algorithms.unsupervised.interfaces import IKMeans


@interface.implementer(IKMeans)
class KMeans(AbstractClusterModel):
    """
    Performs KMeans clustering - super basic.

    Parameters are the point set and number of clusters
    to find.
    """

    def __init__(self, data_frame, **kwargs):
        super(KMeans, self).__init__(data_frame)
        self.cls = SK_KMeans(**kwargs)

    def cluster(self):
        """
        Performs KMeans clustering.

        This algorithm will force the data set into
        the given k clusters, no matter the distribution.
        """
        return self.cls.fit_predict(self._data.to_matrix())

    @property
    def k(self):
        return self.cls.n_clusters
