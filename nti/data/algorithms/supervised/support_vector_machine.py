import _pickle as cPickle
import base64

from sklearn.svm import SVC

from nti.data.algorithms.supervised import SampleDataSet
from nti.data.algorithms.supervised import point_list_to_dataset

class SupportVectorMachine(SVC):
    
    def __init__(self, x_s, y_s, classes=None, training_size = .7):
        if len(x_s) <= 0:
            raise ValueError("Must have at least one point.")
        super(SupportVectorMachine, self).__init__()
        self.data_set = SampleDataSet(x_s, y_s, classes)
        self.training_size = training_size
        self.classes = classes
        self.success_rate = 0.0
    
    def train(self):
        validation_set_size = int(float(self.data_set.get_size() * (1.0 - self.training_size)))
        validation_set = point_list_to_dataset([self.data_set.get_point(i) for i in range(validation_set_size)], multi_class=self.classes)
        training_set = point_list_to_dataset([p for p in self.data_set if p not in validation_set], multi_class=self.classes)
        
        training_X = training_set.get_X_as_numpy()
        training_Y = training_set.get_Y_as_numpy()
        
        self.fit(training_X, training_Y)
        
        correct = 0
        for p in validation_set:
            correct_ans = p.get_actual_answer()
            pred_ans = self.predict([p.get_attributes_as_list()])[0]
            if correct_ans == pred_ans:
                correct += 1
        try:
            self.success_rate = (correct/validation_set_size) * 100
        except ZeroDivisionError:
            pass
    
    def classify(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError("Inputs must be a list")
        numpy_input = [inputs]
        pred_ans = self.predict(numpy_input)[0]
        return pred_ans
        
    def get_success_rate(self):
        return self.success_rate
    
    def get_pickle(self):
        return self
        