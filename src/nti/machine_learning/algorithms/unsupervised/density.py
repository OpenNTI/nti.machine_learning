from nti.machine_learning.algorithms.unsupervised import AbstractClusterModel
 
from nti.machine_learning.algorithms.utils import distance
from nti.machine_learning.algorithms.utils import similarity
from nti.machine_learning.algorithms.utils import mean_distance
from nti.machine_learning.algorithms.utils import entropy
 
class DBScan(AbstractClusterModel):
    """
    Performs the DBScan density clustering algorithm.
     
    Inputs are the point set, the minimum number of points for a cluster,
    and the epsilon value that designates the fine-ness of the clustering.
    """
    def __init__(self, data_frame, min_pts, eps):
        super(DBScan, self).__init__(data_frame)
        # Mark all points as not yet visited
        self._visited = {i:False for i in range(self._data.size())}
        self.min_pts = int(min_pts)
        self.eps = float(eps)
 
    def _find_neighbors(self, point):
        """
        Find the density neighbors of the point. This is the
        bottleneck of this algorithm: you have to visit all points most 
        every time.
        """
        return [i for i in range(self._data.size()) if not self._visited[i] and distance(
            self._data.get_point(i), point) <= self.eps]
 
    def cluster(self):
        """
        Perform DBScan algorithm
         
        At a high level, this algorithm gets a point, and crawls through
        all the other points within the epsilon parameter and until it can't find anymore.
        These points define a cluster. Do this until all points have been visited.
        """
        for i in range(self._data.size()):
            # If we haven't seen this point yet
            if not self._visited[i]:
                self._visited[i] = True
                # Find its neighbors, the ones that aren't visited and are 
                # within the epsilon distance parameter.
                sphere_points = self._find_neighbors(self._data.get_point(i))
                # If there aren't enough points, skip it.
                if len(sphere_points) < self.min_pts:
                    continue
                else:
                    # Create a new cluster, and add the current point
                    new_cluster = self._data.add_cluster()
                    self._data.change_cluster(i, new_cluster)
                    count = 0
                    # Iterate through all our neighbors
                    for j in sphere_points:
                        count += 1
                        # Delete it for optimization reasons
                        q = self._data.get_point(j)
                        if not self._visited[j]:
                            # If we haven't seen this point, find all its
                            # valid neighbors and add them to the list
                            self._visited[j] = True
                            sphere_points_prime = self._find_neighbors(q)
                            sphere_points.extend(
                                [p for p in sphere_points_prime if p not in sphere_points and not self._visited[p]])
                            # Add this point to our current cluster
                            if self._data.get_cluster_for_point(j) == self.NON_MEMBER:
                                self._data.change_cluster(j, new_cluster)
        # Get all the cluster centers to analyze
        # statistics.
        self._data.get_cluster_centers()
        return self._data._data
 
class Entropic(AbstractClusterModel):
    """
    Performs an entropy-based clustering.
      
    Input parameters are the list of points and a 
    beta value signifying the coarseness of clustering.
      
    Entropy is a measure of disorder, so we are trying
    to achieve minimum entropy.
    """
    def __init__(self, data_frame, beta):
        super(Entropic, self).__init__(data_frame)
        self.beta = float(beta)
  
    def _get_similar(self, center, mean_dist, temp_points):
        """
        Get the points that are within the beta value of similarity,
        and add them to the new cluster.
        """
        new_cluster = self._data.add_cluster()
        for p in temp_points:
            if similarity(mean_dist, self._data.get_point(center), self._data.get_point(p)) >= self.beta:
                self._data.change_cluster(p, new_cluster)
  
    def cluster(self):
        """
        Perform the entropy based clustering.
          
        This algorithm is similar to DBScan in that it walks
        through all the points, identifying clusters as it goes.
        It's math heavy, so there are more than one bottleneck.
        """
        temp_points = [i for i in range(self._data.size())]
        # While there are points not in a cluster
        while len(temp_points) > 0:
            # If we only have one, give it its own cluster
            if len(temp_points) == 1:
                c = self._data.add_cluster()
                self._data.change_cluster(temp_points[0], c)
                break
            # Calculate the mean distance. HUUUGGEEEE bottleneck
            points = [self._data.get_point(i) for i in temp_points]
            mean_dist = mean_distance(points)
            # Calculate the entropies of all the points
            entropies = entropy(points, mean_dist)
            # Get the index of the point with the minimum entropy.
            # This is our new cluster center.
            min_ent = entropies.index(min(entropies))
            center = temp_points[min_ent]
            # Get the points similar to this one.
            self._get_similar(center, mean_dist, temp_points)
            # "Remove" points in a cluster
            temp_points = [p for p in temp_points if self._data.get_cluster_for_point(p) == self.NON_MEMBER]
              
        # Calculate the cluster centers.
        self._data.get_cluster_centers()
        return self._data._data
