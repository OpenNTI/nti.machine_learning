from math import sqrt

class Point():
    """
    Point object for learning algorithms
    
    Arguments are a variable length of dimensional values,
    so its adaptable to any data.
    """
    def __init__(self, *args):
        self.dimensions = len(args)
        self.values = []
        for arg in args:
            if not isinstance(arg, (int, float, complex)):
                raise TypeError('Point values must be numeric')
            self.values.append(arg)
        
    def amplitude(self):
        """
        Get the vector amplitude representation of this point
        """
        summation = 0
        for v in self.values:
            summation += v**2
        return sqrt(summation)
    
    def get(self, index):
        """
        Get a particular dimension
        """
        return self.values[index]
    
    def provide_labels(self, labels):
        """
        zips the dimension values with a set of labels
        """
        if len(labels) != len(self.values):
            raise ValueError('Length of label list must be equal to length '
                            'of value list.')
        return dict(zip(labels, self.values))
