from random import randint

from nti.data.algorithms.utils import distance

from nti.data.algorithms.clustering import Cluster
from nti.data.algorithms.clustering import AbstractClusterModel

class KMeans(AbstractClusterModel):
    """
    Performs KMeans clustering - super basic.
    
    Parameters are the point set and number of clusters
    to find.
    """
    def __init__(self, points, cluster_num):
        super(KMeans, self).__init__(points)
        self.cluster_num = int(cluster_num)
        self.clusters = [Cluster(x, self.dims) for x in range(0, self.cluster_num)]
        self._randomized_clusters()
        
    def _init_cluster(self, point, cluster):
        point.cluster = cluster
    
    def _randomized_clusters(self):
        """
        Put all points in random clusters to start with.
        """
        for p in self.points:
            new_cluster = randint(0, self.cluster_num - 1)
            self._init_cluster(p, new_cluster)
            self.clusters[new_cluster].add_point(p)
        for cluster in self.clusters:
            cluster.find_center()
    
    def _get_new_cluster(self, point):
        """
        Gets the closest cluster center to a point
        """
        distances = {cluster.num: distance(cluster.center, point) for cluster in self.clusters}
        return min(distances, key=distances.get)
    
    def _move_clusters(self, point, cluster):
        """
        Moves a point from one cluster to another
        """
        current_cluster = self.clusters[point.cluster]
        current_cluster.find_point_and_remove(point)
        new_cluster = self.clusters[cluster]
        new_cluster.add_point(point)
    
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
            for p in self.points:
                new_cluster = self._get_new_cluster(p)
                if new_cluster != p.cluster:
                    change = True
                self._move_clusters(p, new_cluster)
            for cluster in self.clusters:
                cluster.find_center()
        return self.clusters
    