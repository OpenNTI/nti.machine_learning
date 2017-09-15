#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from nti.machine_learning.interfaces import IModel
from nti.machine_learning.interfaces import IDataSet

from nti.schema.field import Int
from nti.schema.field import Dict
from nti.schema.field import Number
from nti.schema.field import TextLine


class IUnsupervisedModel(IModel):
    """
    Outlines an unsupervised clustering model
    """

    NON_MEMBER = Int(title=u"Non Member Cluster",
                     description=u"Value representing absence of a cluster",
                     default=-1)

    def cluster():
        """
        Performs the clustering actions, returning
        the DataFrame containing the points with an additional
        column dictating the cluster they belong to.
        """


class IUnsupervisedDataSet(IDataSet):
    """
    Outlines the data and structure for an IUnsupervisedModel
    """

    CLUSTER = TextLine(title=u"Cluster",
                       description=u"The cluster column in an unsupervised data set",
                       default=u"cluster",
                       readonly=True)

    dimensions = Int(title=u"Dimensions",
                     description=u"The dimensionality of the data within the data set",
                     default=0)

    clusters = Dict(title=u"Cluster List",
                    description=u"The list of clusters in the data set",
                    default={})

    size = Int(title=u"Size",
               description=u"The size of the data set",
               default=0)

    def change_cluster(index, new_cluster):
        """
        Alters the cluster of a row from one to another
        """

    def add_cluster():
        """
        Add a new cluster to the data set
        """

    def get_cluster_centers():
        """
        Get the current cluster centers
        """

    def get_point(index):
        """
        Gets a point from the frame
        """

    def get_cluster_for_point(index):
        """
        Gets the cluster value for a point
        """

    def get_clusters():
        """
        Gets the cluster sets for a point
        """


class IKMeans(IUnsupervisedModel):
    """
    Represents a KMeans clustering model.
    """

    k = Int(title=u"K",
            description=u"The number of points to find",
            default=2)


class IDBScan(IUnsupervisedModel):
    """
    Represents a DBScan clustering model
    """

    min_pts = Int(title=u"Minimum Points",
                  description=u"The minimum number of points to define a cluster",
                  default=50)

    eps = Number(title=u"Epsilon",
                 description=u"The epsilon tolerance for density",
                 default=0.1)


class IEntropic(IUnsupervisedModel):
    """
    Represents an entropic clustering model
    """

    beta = Number(title=u"Beta",
                  description=u"The beta tolerance for similarity in points",
                  default=0.5)
