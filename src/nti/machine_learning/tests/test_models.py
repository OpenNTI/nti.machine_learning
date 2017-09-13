#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

# disable: accessing protected members, too many methods
# pylint: disable=W0212,R0904

from hamcrest import assert_that

from nti.machine_learning.algorithms import SupportVectorMachine
from nti.machine_learning.algorithms import KMeans
from nti.machine_learning.algorithms import DBScan

from nti.machine_learning.tests import SupervisedLearningLayerTest
from nti.machine_learning.tests import UnsupervisedLearningLayerTest

class TestSupervisedModels(SupervisedLearningLayerTest):
    """
    Test known available supervised models
    """

    def test_svm(self):
        svm = SupportVectorMachine(self.example_frame,
                                   self.example_prediction_columns)
        svm.train()
        assert_that(svm.success_rate, 1.0)

class TestUnsupervisedModels(UnsupervisedLearningLayerTest):
    """
    Test known available unsupervised models
    """

    def test_kmeans(self):
        kmeans = KMeans(self.example_frame, n_clusters=2)
        clusters = kmeans.cluster()
        clusters = set(clusters)
        assert_that(len(clusters), 2)

    def test_dbscan(self):
        dbscan = DBScan(self.example_frame, eps=15, min_samples=1)
        clusters = dbscan.cluster()
        clusters = set(clusters)
        assert_that(len(clusters), 2)
