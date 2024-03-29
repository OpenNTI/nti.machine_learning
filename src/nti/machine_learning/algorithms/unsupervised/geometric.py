#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from sklearn.cluster import KMeans as SK_KMeans

from zope import interface

from nti.machine_learning.algorithms.unsupervised import AbstractClusterModel

from nti.machine_learning.algorithms.unsupervised.interfaces import IKMeans

logger = __import__('logging').getLogger(__name__)


@interface.implementer(IKMeans)
class KMeans(AbstractClusterModel):
    """
    Performs KMeans clustering - super basic.

    Parameters are the point set and number of clusters
    to find.
    """

    def cluster(self, data_frame, **kwargs):
        """
        Performs KMeans clustering.

        This algorithm will force the data set into
        the given k clusters, no matter the distribution.
        """
        super(KMeans, self).cluster(data_frame)
        self.cls = SK_KMeans(**kwargs)
        return self.cls.fit_predict(self._data.to_matrix())

    @property
    def k(self):
        return self.cls.n_clusters
