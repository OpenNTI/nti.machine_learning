#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from random import randint

from zope import interface

from nti.machine_learning.algorithms.unsupervised import AbstractClusterModel

from nti.machine_learning.algorithms.utils import distance

from nti.machine_learning.unsupervised.interfaces import IKMeans

from nti.schema.fieldproperty import createDirectFieldProperties

from nti.schema.schema import SchemaConfigured

class KMeans(AbstractClusterModel,
             SchemaConfigured):
    """
    Performs KMeans clustering - super basic.

    Parameters are the point set and number of clusters
    to find.
    """
    createDirectFieldProperties(IKMeans)

    def __init__(self, data_frame, k):
        super(KMeans, self).__init__(data_frame)
        self._k = int(k)
        for _ in range(self._k):
            self._data.add_cluster()
        self._randomized_clusters()

    def _randomized_clusters(self):
        """
        Put all points in random clusters to start with.
        """
        for i in range(self._data.size()):
            new_cluster = randint(0, self._k - 1)
            self._move_clusters(i, new_cluster)
        self._centers = self._data.get_cluster_centers()
        print(self._centers)

    def _get_new_cluster(self, index):
        """
        Gets the closest cluster center to a point
        """
        distances = {
            c: distance(self._centers[c], self._data.get_point(index)) for c in self._centers.keys()
        }
        return min(distances, key=distances.get)

    def _move_clusters(self, index, cluster):
        """
        Moves a point from one cluster to another
        """
        self._data.change_cluster(index, cluster)

    def cluster(self):
        """
        Performs KMeans clustering.

        This algorithm will force the data set into 
        the given k clusters, no matter the distribution.
        """
        change = True
        # While changes have been made
        while change:
            change = False
            # Find the closest cluster center to this point
            # and move the point to that cluster.
            for i in range(self._data.size()):
                new_cluster = self._get_new_cluster(i)
                if new_cluster != self._data.get_cluster_for_point(i):
                    change = True
                    self._move_clusters(i, new_cluster)
            self.centers = self._data.get_cluster_centers()
        return self._data
