from nti.data.algorithms.common import Point

class AbstractClusterModel():
    """
    Serves as a base for a clustering model.
    
    Takes a set of points, determines the dimensions it is clustering,
    and marks all points as not yet belonging to any cluster.
    """
    
    NON_MEMBER = -1
    
    def __init__(self, points):
        if len(points) <= 1:
            raise ValueError('Points list length must be > 1')
        self.dims = points[0].dimensions
        self.points = points
        for p in self.points:
            p.cluster = self.NON_MEMBER
            
    def cluster(self):
        """
        Function that performs the clustering.
        """
        raise NotImplementedError('cluster method must be provided.')

class Cluster():
    """
    Represent a cluster object.
    """
    
    def __init__(self, num, dimensions):
        self.points = []
        self.num = num
        self.dims = dimensions

    def add_point(self, point):
        """
        Add a point to this cluster, and mark the point
        as belonging to the cluster
        """
        point.cluster = self.num
        self.points.append(point)

    def find_center(self):
        """
        Calculate the center of the cluster.
        """
        if len(self.points) <= 0:
            zeroes = [0 for i in range(self.dims)]
            # If there are no points, mark center as zero.
            self.center = Point(*zeroes)
            return
        center_vals = [sum([p.get(i) for p in self.points]) / len(self.points) for i in range(0, self.dims)]
        self.center = Point(*center_vals)

    def find_point_and_remove(self, point):
        """
        Remove a point object from the cluster
        """
        index = self.points.index(point)
        del self.points[index]
        
    def length(self):
        """
        Get the number of points in the cluster
        """
        return len(self.points)
