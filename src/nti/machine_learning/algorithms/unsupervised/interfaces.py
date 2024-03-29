#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from nti.machine_learning.interfaces import IModel
from nti.machine_learning.interfaces import IDataSet

from nti.schema.field import Int
from nti.schema.field import Dict
from nti.schema.field import TextLine

logger = __import__('logging').getLogger(__name__)


class IUnsupervisedModel(IModel):
    """
    Outlines an unsupervised clustering model
    """

    NON_MEMBER = Int(title=u"Non Member Cluster",
                     description=u"Value representing absence of a cluster",
                     default=-1)

    def cluster(data_fame):
        """
        Performs the clustering actions, returning
        the DataFrame containing the points with an additional
        column dictating the cluster they belong to.

        data_frame:
            A pandas data frame containing the data
            to be clustered
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


class IDBScan(IUnsupervisedModel):
    """
    Represents a DBScan clustering model
    """
