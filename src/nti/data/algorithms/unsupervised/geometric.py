from random import randint

from nti.data.algorithms.utils import distance

from nti.data.algorithms.unsupervised import AbstractClusterModel

class KMeans(AbstractClusterModel):
    """
    Performs KMeans clustering - super basic.
    
    Parameters are the point set and number of clusters
    to find.
    """
    def __init__(self, data_frame, cluster_num):
        super(KMeans, self).__init__(data_frame)
        self._cluster_num = int(cluster_num)
        for i in range(self._cluster_num):
            self._data.add_cluster()
        self._randomized_clusters()
    
    def _randomized_clusters(self):
        """
        Put all points in random clusters to start with.
        """
        for i in range(self._data.size()):
            new_cluster = randint(0, self._cluster_num - 1)
            self._move_clusters(i, new_cluster)
        self._centers = self._data.get_cluster_centers()
        print(self._centers)
    
    def _get_new_cluster(self, index):
        """
        Gets the closest cluster center to a point
        """
        distances = {c: distance(self._centers[c], self._data.get_point(index)) for c in self._centers.keys()}
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
    