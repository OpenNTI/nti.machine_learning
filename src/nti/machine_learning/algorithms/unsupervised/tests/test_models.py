#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

# disable: accessing protected members, too many methods
# pylint: disable=W0212,R0904

from hamcrest import assert_that

from nti.testing.matchers import validly_provides

from nti.machine_learning.algorithms.unsupervised.density import DBScan

from nti.machine_learning.algorithms.unsupervised.geometric import KMeans

from nti.machine_learning.algorithms.unsupervised.interfaces import IDBScan
from nti.machine_learning.algorithms.unsupervised.interfaces import IKMeans

from nti.machine_learning.tests import UnsupervisedLearningLayerTest


class TestUnsupervisedModels(UnsupervisedLearningLayerTest):
    """
    Test known available unsupervised models
    """

    def test_kmeans(self):
        kmeans = KMeans(self.example_frame, n_clusters=2)
        assert_that(kmeans, validly_provides(IKMeans))
        clusters = kmeans.cluster()
        clusters = set(clusters)
        assert_that(len(clusters), 2)

    def test_dbscan(self):
        dbscan = DBScan(self.example_frame, eps=15, min_samples=1)
        assert_that(dbscan, validly_provides(IDBScan))
        clusters = dbscan.cluster()
        clusters = set(clusters)
        assert_that(len(clusters), 2)
