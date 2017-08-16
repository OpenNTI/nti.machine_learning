from nti.data.algorithms.clustering import Cluster
from nti.data.algorithms.clustering import AbstractClusterModel

from nti.data.algorithms.utils import distance
from nti.data.algorithms.utils import similarity
from nti.data.algorithms.utils import mean_distance
from nti.data.algorithms.utils import entropy

class DBScan(AbstractClusterModel):
    """
    Performs the DBScan density clustering algorithm.
    
    Inputs are the point set, the minimum number of points for a cluster,
    and the epsilon value that designates the fine-ness of the clustering.
    """
    def __init__(self, points, min_pts, eps):
        super(DBScan, self).__init__(points)
        # Mark all points as not yet visited
        for p in self.points:
            p.visited = False
        self.min_pts = int(min_pts)
        self.eps = float(eps)
        self.clusters = []

    def _find_neighbors(self, point):
        """
        Find the density neighbors of the point. This is the
        bottleneck of this algorithm: you have to visit all points most 
        every time.
        """
        return [p for p in self.points if not p.visited and distance(
            p, point) <= self.eps]

    def cluster(self):
        """
        Perform DBScan algorithm
        
        At a high level, this algorithm gets a point, and crawls through
        all the other points within the epsilon parameter and until it can't find anymore.
        These points define a cluster. Do this until all points have been visited.
        """
        for p in self.points:
            # If we haven't seen this point yet
            if not p.visited:
                p.visited = True
                # Find its neighbors, the ones that aren't visited and are 
                # within the epsilon distance parameter.
                sphere_points = self._find_neighbors(p)
                # If there aren't enough points, skip it.
                if len(sphere_points) < self.min_pts:
                    continue
                else:
                    # Create a new cluster, and add the current point
                    cluster = Cluster(len(self.clusters), self.dims)
                    cluster.add_point(p)
                    self.clusters.append(cluster)
                    # Iterate through all our neighbors
                    while sphere_points:
                        # Delete it for optimization reasons
                        q = sphere_points.pop()
                        if not q.visited:
                            # If we haven't seen this point, find all its
                            # valid neighbors and add them to the list
                            q.visited = True
                            sphere_points_prime = self._find_neighbors(q)
                            sphere_points.extend(
                                [p for p in sphere_points_prime if p not in sphere_points and not p.visited])
                            # Add this point to our current cluster
                            if q.cluster == self.NON_MEMBER:
                                cluster.add_point(q)
        # Get all the cluster centers to analyze
        # statistics.
        for c in self.clusters:
            c.find_center()
        return self.clusters

class Entropic(AbstractClusterModel):
    """
    Performs an entropy-based clustering.
    
    Input parameters are the list of points and a 
    beta value signifying the coarseness of clustering.
    
    Entropy is a measure of disorder, so we are trying
    to achieve minimum entropy.
    """
    def __init__(self, points, beta):
        super(Entropic, self).__init__(points)
        self.beta = beta
        self.clusters = []

    def _get_similar(self, center, mean_dist, temp_points):
        """
        Get the points that are within the beta value of similarity,
        and add them to the new cluster.
        """
        cluster = Cluster(len(self.clusters), self.dims)
        for p in temp_points:
            if similarity(mean_dist, center, p) >= self.beta:
                cluster.add_point(p)
        self.clusters.append(cluster)

    def cluster(self):
        """
        Perform the entropy based clustering.
        
        This algorithm is similar to DBScan in that it walks
        through all the points, identifying clusters as it goes.
        It's math heavy, so there are more than one bottleneck.
        """
        temp_points = self.points
        # While there are points not in a cluster
        while len(temp_points) > 0:
            # If we only have one, give it its own cluster
            if len(temp_points) == 1:
                c = Cluster(len(self.clusters), self.dims)
                c.add_point(temp_points[0])
                self.clusters.append(c)
                break
            # Calculate the mean distance. HUUUGGEEEE bottleneck
            mean_dist = mean_distance(temp_points)
            # Calculate the entropies of all the points
            entropies = entropy(temp_points, mean_dist)
            # Get the index of the point with the minimum entropy.
            # This is our new cluster center.
            min_ent = entropies.index(min(entropies))
            center = temp_points[min_ent]
            # Get the points similar to this one.
            self._get_similar(center, mean_dist, temp_points)
            # "Remove" points in a cluster
            temp_points = [p for p in self.points if p.cluster == self.NON_MEMBER]
            
        # Calculate the cluster centers.
        for c in self.clusters:
            c.find_center()
        return self.clusters
