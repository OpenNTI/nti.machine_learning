"""
Used by the demonstrate command. builds fake data to cluster
and plots it.
"""

from matplotlib import pyplot
from matplotlib import transforms
from matplotlib import lines

from mpl_toolkits.mplot3d import Axes3D

from random import randint

from nti.data.algorithms.common import Point
from nti.data.algorithms.common import SupervisedPoint

class Functions():
    
    @staticmethod
    def LINEAR(x):
        return x
    
    @staticmethod
    def QUADRATIC(x):
        return x**2
    
class DataGenerator():
    """
    Gets a random set of data points either uniform 
    or non-uniform.
    """
    
    
    def __init__(self, point_count):
        self.point_count = int(point_count)

    def _get_point(self):
        values = [randint(0, 100) for i in range(0, 3)]
        return Point(*values)

    def _get_nonuniform_point(self, upper=100, lower=0):
        dims = []
        dims.append(randint(lower, upper))
        dims.append(randint(0, 100))
        dims.append(randint(0, 100))
        return Point(*dims)

    def _get_data(self, non_uniform):
        """
        Get the points. if non-uniform, present
        in two groups with obvious split in the middle.
        """
        if non_uniform:
            points = [
                self._get_nonuniform_point(
                    upper=30) for i in range(
                    0, self.point_count // 2)]
            points.extend([self._get_nonuniform_point(lower=70)
                           for i in range(0, self.point_count // 2)])
            return points
        return [self._get_point() for i in range(0, self.point_count)]

    def generate_sample_data(self, non_uniform=False):
        return self._get_data(non_uniform)
    
    def generate_with_function(self, function, upper_x, stddev):
        points = []
        for i in range(self.point_count):
            x = randint(0, upper_x)
            q = randint(0, stddev) - stddev / 2
            sup_vec = 5 if q >= 0 else -5
            func_out = function(x)
            points.append(Point(*[x, func_out + q + sup_vec])) 
        return points

class SupervisedGenerator(DataGenerator):
    
    def generate_sample_data(self):
        points = self._get_data(True)
        sup_points = []
        for point in points:
            sup_point = SupervisedPoint(*point.values)
            sup_point.correct = 0 if sup_point.get(0) < 50 else 1
            sup_points.append(sup_point)
        return sup_points

class Plotter():
    """
    Plot the given data.
    """
    def __init__(self, points):
        self.x = [p.get(0) for p in points]
        self.y = [p.get(1) for p in points]
        self.z = [p.get(2) for p in points]
        self.cluster = [p.cluster for p in points]
        self.fig = pyplot.figure()
        self.ax = Axes3D(self.fig)

    def plot(self, title):
        self.ax.scatter(self.x, self.y, self.z, c=self.cluster)
        pyplot.suptitle(title)
        pyplot.show()
    
class Plotter2D():
    """
    Plot 2 dimensional data
    """
    def __init__(self, points, correct):
        self.x = [p.get(0) for p in points]
        self.y = [p.get(1) for p in points]
        self.val = correct
        
    def plot(self, title):
        _, ax = pyplot.subplots()
        ax.scatter(self.x, self.y, c='black')
        pyplot.show()
