from nti.data import Model
from nti.data import AbstractDataSet

class AbstractClusterModel(Model):
    """
    Serves as a base for a clustering model.
    
    Takes a set of points, determines the dimensions it is clustering,
    and marks all points as not yet belonging to any cluster.
    """
    
    NON_MEMBER = -1
    
    def __init__(self, data_frame):
        if len(data_frame.index.values) <= 1:
            raise ValueError('Points list length must be > 1')
        self._dimensions = len(data_frame.columns)
        self._data = UnsupervisedDataSet(data_frame)
            
    def cluster(self):
        """
        Function that performs the clustering.
        """
        raise NotImplementedError('cluster method must be provided.')

class UnsupervisedDataSet(AbstractDataSet):
    
    _CLUSTER = "cluster"
    
    def __init__(self, data_frame):
        self._data = data_frame
        self._data[self._CLUSTER] = AbstractClusterModel.NON_MEMBER
        self._dimensions = len(self._data.columns) - 1
        self._clusters = {}
        self._size = len(self._data.index.values)
        self._idx = 0
    
    def change_cluster(self, index, new_cluster):
        self._data.set_value(index, self._CLUSTER, new_cluster)
        
    def add_cluster(self):
        new_index = len(self._clusters)
        self._clusters[new_index] = None
        return new_index

    def _get_cluster(self, cluster):
        return self._data.loc[self._data[self._CLUSTER] == cluster].as_matrix()

    def get_cluster_centers(self):
        for c in self._clusters.keys():
            points = self._get_cluster(c)
            if points is None:
                self._clusters[c] = [0 for i in range(self._dimensions)]
            self._clusters[c] = [sum([p[i] for p in points]) / len(points) for i in range(self._dimensions)]
        return self._clusters
    
    def get_point(self, index):
        return self._data.iloc[index,:-1].as_matrix()
    
    def get_cluster_for_point(self, index):
        return self._data.iloc[index,-1]
    
    def size(self):
        return self._size
    
    def get_clusters(self):
        vals = self._data[self._CLUSTER].unique()
        results = []
        for c in vals:
            results.append(self._data[self._data[self._CLUSTER] == c])
        return results
    
    def __iter__(self):
        return iter(self._data.index.values)
    
    def __next__(self):
        self._idx += 1
        if self._idx == self._size:
            self._idx = 0
            raise StopIteration
        return self.get_point(self._idx)
        
