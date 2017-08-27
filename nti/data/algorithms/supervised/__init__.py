from numpy import array

from nti.data.algorithms.common import Point

class SampleDataPoint(Point):
    
    def __init__(self, attributes, prediction):
        super(SampleDataPoint, self).__init__(*attributes)
        self.prediction = prediction
    
    def get_attributes_as_list(self):
        return self.values
    
    def get_actual_answer(self):
        return self.prediction

class SampleDataSet():
    
    _categorical_numeric = []
    _categorical_dict = {}
    
    def __init__(self, x_s, y_s, multi_class=None, as_batch=False):
        if len(x_s) != len(y_s):
            raise ValueError("X and Y must be the same length")
        if multi_class is not None:
            self._categorical_numeric = list(range(0, len(multi_class)))
            self._categorical_dict = dict(zip(multi_class, self._categorical_numeric))
        self.sample_points = []
        for i in range(0, len(x_s)):
            self.sample_points.append(SampleDataPoint(x_s[i], y_s[i]))
        self.multi_class = multi_class
        self.idx = 0
    
    def _get_buckets(self):
        indxs = [i for i in range(len(self.sample_points))]
    
    def is_multi_classification(self):
        return self.multi_class is not None
    
    def get_point(self, i):
        return self.sample_points[i]
    
    def get_size(self):
        return len(self.sample_points)

    def get_X_as_numpy(self):
        X = []
        for point in self.sample_points:
            X.append(point.get_attributes_as_list())
        return array(X)
    
    def get_Y_as_numpy(self, rank=False):
        Y = []
        for point in self.sample_points:
            p = point.get_actual_answer()
            p = [p] if rank else p
            Y.append(p)
        return array(Y)

    def get_categorical(self, inp):
        return self._categorical_dict[inp]
    
    def __iter__(self):
        return iter(self.sample_points)
    
    def __next__(self):
        self.idx += 1
        try:
            return self.sample_points[self.idx]
        except IndexError:
            self.idx = 0
            raise StopIteration

def point_list_to_dataset(point_list, multi_class=None):
    x_s = [p.get_attributes_as_list() for p in point_list]
    y_s = [p.get_actual_answer() for p in point_list]
    return SampleDataSet(x_s, y_s, multi_class=multi_class)
